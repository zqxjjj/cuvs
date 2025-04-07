/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "common.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <optional>
#include <fstream>
#include <sstream>
#include <string>

#include <cuda_fp16.h>

// support cuda stream
#include <thread>
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>

// support called from python
// #include <torch/extension.h>
// #include <cuda_bf16.h>
// #include <ATen/ATen.h>
// #include <ATen/Context.h>
// #include <ATen/cuda/CUDAContext.h>

/**
 * @brief CUDA kernel to convert float values to half precision
 */
__global__ void convert_float_to_half(const float* input, __half* output, size_t total_elements)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    output[idx] = __float2half(input[idx]);
  }
}

/**
 * @brief Print device matrix content to console
 */
void print_device_matrix(const raft::device_resources& handle,
                         const raft::device_vector_view<uint32_t, int64_t>& matrix)
{
  auto n_rows = matrix.extent(0);
  
  std::vector<uint32_t> host_data(n_rows);
  
  raft::update_host(host_data.data(), matrix.data_handle(), n_rows, raft::resource::get_cuda_stream(handle));
  
  raft::resource::sync_stream(handle);
  
  for (uint32_t i = 0; i < n_rows; i++) {
      std::cout << host_data[i] << " ";
  }
  std::cout << std::endl;
}

/**
 * @brief Load data from a CSV file into a host matrix
 */
raft::host_matrix<float, int64_t> load_csv(const raft::resources& handle, 
                                          const std::string& filepath, 
                                          int64_t start_row,
                                          int64_t n_rows, 
                                          int64_t n_cols) {
    auto host_data = raft::make_host_matrix<float, int64_t>(n_rows, n_cols);
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }
    
    // Skip rows if needed
    std::string line;
    int64_t current_row = 0;
    while (current_row < start_row && std::getline(file, line)) {
        current_row++;
        if (current_row % 50000 == 0) std::cout << "Skipped " << current_row << " rows" << std::endl;
    }
    
    // Read data line by line
    int64_t row = 0;
    while (std::getline(file, line) && row < n_rows) {
        std::stringstream ss(line);
        std::string cell;
        int64_t col = 0;
        while (std::getline(ss, cell, ',') && col < n_cols) {
            host_data.data_handle()[row * n_cols + col] = std::stof(cell);
            col++;
        }
        row++;
        if (row % 10000 == 0) std::cout << "Loaded " << row << " rows from CSV (starting from row " << start_row << ")" << std::endl;
    }
    
    std::cout << "Loaded " << row << " rows from CSV (starting from row " << start_row << ")" << std::endl;
    return host_data;
}

/**
 * @brief Build global segment index
 */
void build_segment_global(raft::device_resources const& dev_resources,
                         cuvs::neighbors::ivf_flat::index<half, int64_t>& index,
                         uint16_t* keys,
                         int seq_len,
                         int n_clusters)
{
  using namespace cuvs::neighbors;
  auto stream = raft::resource::get_cuda_stream(dev_resources);
  
  auto keys_ptr = reinterpret_cast<half*>(keys);
  auto keys_view_host = raft::make_mdspan<half, int64_t>(keys_ptr, raft::make_extents<int64_t>(seq_len, 128));
  
  auto dataset_keys = raft::make_device_matrix<half, int64_t>(dev_resources, seq_len, 128);
  raft::copy(dataset_keys.data_handle(), keys_view_host.data_handle(), seq_len * 128, stream);
  raft::resource::sync_stream(dev_resources, stream);
  
  auto build_params = ivf_flat::index_params();
  build_params.n_lists = n_clusters;
  build_params.segment_build = true;
  
  ivf_flat::build(dev_resources, build_params, raft::make_const_mdspan(dataset_keys.view()), index);

  int64_t n_rows = dataset_keys.extent(0);
  raft::device_vector<uint32_t, int64_t> new_labels = raft::make_device_mdarray<uint32_t>(
    dev_resources, raft::resource::get_large_workspace_resource(dev_resources), raft::make_extents<int64_t>(n_rows)); 
  
  ivf_flat::compute_labels(dev_resources, &index, raft::make_const_mdspan(dataset_keys.view()), new_labels, n_rows);
}

/**
 * @brief Build global segment index with multiple streams for parallelization
 */
void build_segment_global_multistream(raft::device_resources const& dev_resources,
                                     std::vector<cuvs::neighbors::ivf_flat::index<half,int64_t>*>& indices,
                                     std::vector<uint16_t*>& keys_list,
                                     std::vector<int>& seq_lengths,
                                     std::vector<int>& n_clusters_list)
{
  int num_streams = indices.size();
  // Create a CUDA stream pool
  raft::resource::set_cuda_stream_pool(dev_resources, std::make_shared<rmm::cuda_stream_pool>(num_streams));
  
  // Vector to store threads
  std::vector<std::thread> threads;
  threads.reserve(num_streams);
  
  // Launch a thread for each dataset/index pair
  for (int i = 0; i < num_streams; i++) {
    threads.emplace_back([&dev_resources, &indices, &keys_list, &seq_lengths, &n_clusters_list, i]() {
      // Get a stream from the pool
      auto stream = raft::resource::get_next_usable_stream(dev_resources, i);
      
      // Create a device_resources object with the stream
      raft::device_resources stream_pool_handle(dev_resources);
      raft::resource::set_cuda_stream(stream_pool_handle, stream);
      
      // Perform build_segment_global operations for this dataset
      using namespace cuvs::neighbors;
      
      auto keys_ptr = reinterpret_cast<half*>(keys_list[i]);
      auto keys_view_host = raft::make_mdspan<half, int64_t>(keys_ptr, 
                                             raft::make_extents<int64_t>(seq_lengths[i], 128));
      
      auto dataset_keys = raft::make_device_matrix<half, int64_t>(stream_pool_handle, 
                                                 seq_lengths[i], 128);
      raft::copy(dataset_keys.data_handle(), keys_view_host.data_handle(), 
                 seq_lengths[i] * 128, stream);
      raft::resource::sync_stream(stream_pool_handle, stream);
      
      // Set params - note that segment_build is set to true
      auto build_params = ivf_flat::index_params();
      build_params.n_lists = n_clusters_list[i];
      build_params.segment_build = true;
      
      // Build index
      ivf_flat::build(stream_pool_handle, build_params, 
                      raft::make_const_mdspan(dataset_keys.view()), *indices[i]);
      
      // After building the index, compute labels for the dataset
      int64_t n_rows = dataset_keys.extent(0);
      raft::device_vector<uint32_t, int64_t> new_labels = raft::make_device_vector<uint32_t, int64_t>(
        stream_pool_handle, n_rows);
      
      ivf_flat::compute_labels(stream_pool_handle, indices[i], 
                             raft::make_const_mdspan(dataset_keys.view()), 
                             new_labels, n_rows);
    });
  }

  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }
}

/**
 * @brief Creates and returns a new IVF-FLAT index
 */
cuvs::neighbors::ivf_flat::index<half, int64_t>* get_index(raft::device_resources const& dev_resources)
{
  cuvs::neighbors::ivf_flat::index_params params = cuvs::neighbors::ivf_flat::index_params();
  return new cuvs::neighbors::ivf_flat::index<half, int64_t>(dev_resources, params, 128);
}

/**
 * @brief Build local segment index
 */
void build_segment_local(raft::device_resources const& dev_resources,
                         cuvs::neighbors::ivf_flat::index<half, int64_t>& index,
                         uint16_t* keys,
                         int seq_len,
                         int n_clusters)
{
  using namespace cuvs::neighbors;
  auto stream = raft::resource::get_cuda_stream(dev_resources);
  
  auto keys_ptr = reinterpret_cast<half*>(keys);
  auto keys_view_host = raft::make_mdspan<half, int64_t>(keys_ptr, raft::make_extents<int64_t>(seq_len, 128));
  
  auto dataset_keys = raft::make_device_matrix<half, int64_t>(dev_resources, seq_len, 128);
  raft::copy(dataset_keys.data_handle(), keys_view_host.data_handle(), seq_len * 128, stream);
  raft::resource::sync_stream(dev_resources, stream);
  
  auto build_params = ivf_flat::index_params();
  build_params.n_lists = n_clusters;
  build_params.segment_build = true;
  
  ivf_flat::build(dev_resources, build_params, raft::make_const_mdspan(dataset_keys.view()), index);
}

/**
 * @brief Build local segment index with multiple streams for parallelization
 */
void build_segment_local_multistream(raft::device_resources const& dev_resources,
                                     std::vector<cuvs::neighbors::ivf_flat::index<half,int64_t>*>& indices,
                                     std::vector<uint16_t*>& keys_list,
                                     std::vector<int>& seq_lengths,
                                     std::vector<int>& n_clusters_list)
{
  int num_streams = indices.size();
  // Create a CUDA stream pool
  raft::resource::set_cuda_stream_pool(dev_resources, std::make_shared<rmm::cuda_stream_pool>(num_streams));
  
  // Vector to store threads
  std::vector<std::thread> threads;
  threads.reserve(num_streams);
  
  // Launch a thread for each dataset/index pair
  for (int i = 0; i < num_streams; i++) {
    threads.emplace_back([&dev_resources, &indices, &keys_list, &seq_lengths, &n_clusters_list, i]() {
      // Get a stream from the pool
      auto stream = raft::resource::get_next_usable_stream(dev_resources, i);
      
      // Create a device_resources object with the stream
      raft::device_resources stream_pool_handle(dev_resources);
      raft::resource::set_cuda_stream(stream_pool_handle, stream);
      
      // Perform build_segment_local operations for this dataset
      using namespace cuvs::neighbors;
      
      auto keys_ptr = reinterpret_cast<half*>(keys_list[i]);
      auto keys_view_host = raft::make_mdspan<half, int64_t>(keys_ptr, 
                                             raft::make_extents<int64_t>(seq_lengths[i], 128));
      
      auto dataset_keys = raft::make_device_matrix<half, int64_t>(stream_pool_handle, 
                                                 seq_lengths[i], 128);
      raft::copy(dataset_keys.data_handle(), keys_view_host.data_handle(), 
                 seq_lengths[i] * 128, stream);
      raft::resource::sync_stream(stream_pool_handle, stream);
      
      // Set params - note the segment_build = true which is different from build_global
      auto build_params = ivf_flat::index_params();
      build_params.n_lists = n_clusters_list[i];
      build_params.segment_build = true;
      
      // Build index
      ivf_flat::build(stream_pool_handle, build_params, 
                      raft::make_const_mdspan(dataset_keys.view()), *indices[i]);
    });
  }

  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }
}

/**
 * @brief Build global index
 */
void build_global(raft::device_resources const& dev_resources, 
                  cuvs::neighbors::ivf_flat::index<half,int64_t>& idx,
                  uint16_t* keys, // gpu pointer
                  int seq_len,
                  int n_clusters)
{
  using namespace cuvs::neighbors;
  auto stream = raft::resource::get_cuda_stream(dev_resources);
  // casting
  auto keys_ptr = reinterpret_cast<half*>(keys);
  auto keys_view_host = raft::make_mdspan<half, int64_t>(keys_ptr, raft::make_extents<int64_t>(seq_len, 128));
  // prepare keys to be built
  auto dataset_keys = raft::make_device_matrix<half, int64_t>(dev_resources, seq_len, 128);
  raft::copy(dataset_keys.data_handle(), keys_view_host.data_handle(), seq_len * 128, stream);
  raft::resource::sync_stream(dev_resources, stream);
  
  // set params
  auto build_params = ivf_flat::index_params();
  build_params.n_lists = n_clusters;
  // build index
  ivf_flat::build(dev_resources, build_params, raft::make_const_mdspan(dataset_keys.view()), idx);
}

/**
 * @brief Multistream version of build_global
 */
void build_global_multistream(raft::device_resources const& dev_resources,
                              std::vector<cuvs::neighbors::ivf_flat::index<half,int64_t>*>& indices,
                              std::vector<uint16_t*>& keys_list,
                              std::vector<int>& seq_lengths,
                              std::vector<int>& n_clusters_list)
{
  int num_streams = indices.size();
  // Create a CUDA stream pool
  raft::resource::set_cuda_stream_pool(dev_resources, std::make_shared<rmm::cuda_stream_pool>(num_streams));
  
  // Vector to store threads
  std::vector<std::thread> threads;
  threads.reserve(num_streams);
  
  // Launch a thread for each dataset/index pair
  for (int i = 0; i < num_streams; i++) {
    threads.emplace_back([&dev_resources, &indices, &keys_list, &seq_lengths, &n_clusters_list, i]() {
      // Get a stream from the pool
      auto stream = raft::resource::get_next_usable_stream(dev_resources, i);
      
      // Create a device_resources object with the stream
      raft::device_resources stream_pool_handle(dev_resources);
      raft::resource::set_cuda_stream(stream_pool_handle, stream);
      
      // Perform build_global operations for this dataset
      using namespace cuvs::neighbors;
      
      auto keys_ptr = reinterpret_cast<half*>(keys_list[i]);
      auto keys_view_host = raft::make_mdspan<half, int64_t>(keys_ptr, 
                                             raft::make_extents<int64_t>(seq_lengths[i], 128));
      
      auto dataset_keys = raft::make_device_matrix<half, int64_t>(stream_pool_handle, 
                                                 seq_lengths[i], 128);
      raft::copy(dataset_keys.data_handle(), keys_view_host.data_handle(), 
                 seq_lengths[i] * 128, stream);
      raft::resource::sync_stream(stream_pool_handle, stream);
      
      // Set params
      auto build_params = ivf_flat::index_params();
      build_params.n_lists = n_clusters_list[i];
      
      // Build index
      ivf_flat::build(stream_pool_handle, build_params, 
                      raft::make_const_mdspan(dataset_keys.view()), *indices[i]);
    });
  }

  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }
}

// test build_segment_local and build_segment_local_multistream
int main()
{
  using namespace cuvs::neighbors;
  raft::device_resources dev_resources;
  
  // Set pool memory resource with 12 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 12ull * 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);
  
  // Define dataset dimensions
  int64_t n_dim = 128;
  int64_t n_samples_per_head = 130293; // Number of vectors per head
  int n_clusters = 4096; // Number of clusters for IVF-Flat
  int n_heads = 8; // Total number of heads to process
  
  // CSV file has shape (1042344, 128), which is approximately 8 heads of 130293 vectors each
  std::string csv_path = "/home/v-xle/cuvs/examples/cpp/src/csv-data/key_states_layer_10.csv";
  
  auto stream = raft::resource::get_cuda_stream(dev_resources);
  
  // Vectors to store per-head data
  std::vector<raft::device_matrix<float, int64_t>> fp32_datasets;
  std::vector<raft::device_matrix<__half, int64_t>> fp16_datasets;
  std::vector<uint16_t*> keys_ptrs(n_heads);
  std::vector<int> seq_lengths(n_heads, n_samples_per_head);
  std::vector<int> n_clusters_list(n_heads, n_clusters);
  
  // Reserve space for the vectors
  fp32_datasets.reserve(n_heads);
  fp16_datasets.reserve(n_heads);
  
  std::cout << "Loading dataset from CSV..." << std::endl;
  
  // Load all heads from CSV
  for (int head = 0; head < n_heads; head++) {
    // Load data for this head
    auto host_dataset_fp32 = load_csv(dev_resources, csv_path, head * n_samples_per_head, 
                                      n_samples_per_head, n_dim);
    
    // Create device matrices for this head
    fp32_datasets.push_back(raft::make_device_matrix<float, int64_t>(dev_resources, n_samples_per_head, n_dim));
    fp16_datasets.push_back(raft::make_device_matrix<__half, int64_t>(dev_resources, n_samples_per_head, n_dim));
    
    // Copy host data to device
    raft::copy(fp32_datasets[head].data_handle(), host_dataset_fp32.data_handle(), 
               n_samples_per_head * n_dim, stream);
    
    // Convert FP32 to FP16
    size_t total_elements = n_samples_per_head * n_dim;
    int threadsPerBlock = 256;
    int blocksDataset = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    convert_float_to_half<<<blocksDataset, threadsPerBlock>>>(
      fp32_datasets[head].data_handle(), fp16_datasets[head].data_handle(), total_elements);
    cudaDeviceSynchronize();
    
    // Store the keys pointer
    keys_ptrs[head] = reinterpret_cast<uint16_t*>(fp16_datasets[head].data_handle());
    
    std::cout << "Head " << head + 1 << " of " << n_heads << " loaded and converted to FP16" << std::endl;
  }
  
  // ---------- Serial processing with build_segment_local ----------
  std::cout << "\nTesting build_segment_local (serial processing):" << std::endl;
  
  // Create indices for serial processing
  std::vector<cuvs::neighbors::ivf_flat::index<half, int64_t>*> serial_indices(n_heads);
  for (int i = 0; i < n_heads; i++) {
    serial_indices[i] = get_index(dev_resources);
  }
  
  // Start timing for serial processing
  auto serial_start = std::chrono::high_resolution_clock::now();
  
  // Process each head serially
  for (int head = 0; head < n_heads; head++) {
    build_segment_local(dev_resources, *serial_indices[head], 
                        keys_ptrs[head], n_samples_per_head, n_clusters);
    std::cout << "Head " << head + 1 << " processed serially" << std::endl;
  }
  
  // End timing for serial processing
  auto serial_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> serial_elapsed = serial_end - serial_start;
  
  std::cout << "Serial processing completed successfully" << std::endl;
  std::cout << "Total serial build time: " << serial_elapsed.count() << " ms" << std::endl;
  
  // ---------- Parallel processing with build_segment_local_multistream ----------
  std::cout << "\nTesting build_segment_local_multistream (parallel processing):" << std::endl;
  
  // Create indices for parallel processing
  std::vector<cuvs::neighbors::ivf_flat::index<half, int64_t>*> parallel_indices(n_heads);
  for (int i = 0; i < n_heads; i++) {
    parallel_indices[i] = get_index(dev_resources);
  }
  
  // Start timing for parallel processing
  auto parallel_start = std::chrono::high_resolution_clock::now();
  
  // Process all heads in parallel
  build_segment_local_multistream(dev_resources, parallel_indices, keys_ptrs, seq_lengths, n_clusters_list);
  
  // End timing for parallel processing
  auto parallel_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> parallel_elapsed = parallel_end - parallel_start;
  
  std::cout << "Parallel processing completed successfully" << std::endl;
  std::cout << "Total parallel build time: " << parallel_elapsed.count() << " ms" << std::endl;
  
  // Speedup calculation
  double speedup = serial_elapsed.count() / parallel_elapsed.count();
  std::cout << "\nSpeedup from parallelization: " << speedup << "x" << std::endl;
  
  // Clean up resources
  for (int i = 0; i < n_heads; i++) {
    ivf_flat::helpers::reset_index(dev_resources, serial_indices[i]);
    delete serial_indices[i];
    
    ivf_flat::helpers::reset_index(dev_resources, parallel_indices[i]);
    delete parallel_indices[i];
  }
  return 0;
}