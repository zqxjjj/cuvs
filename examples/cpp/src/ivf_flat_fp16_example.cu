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

// connect with python
// #include <pybind11/pybind11.h>
// #include <torch/extension.h>

__global__ void convert_float_to_half(const float* input, __half* output, size_t total_elements)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    output[idx] = __float2half(input[idx]);
  }
}


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

// Function to load data from a CSV file into a host matrix
raft::host_matrix<float, int64_t> load_csv(const raft::resources& handle, 
                                          const std::string& filepath, 
                                          int64_t start_row,
                                          int64_t n_rows, 
                                          int64_t n_cols) {
    // Create host matrix
    auto host_data = raft::make_host_matrix<float, int64_t>(n_rows, n_cols);
    
    // Open CSV file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }
    
    // Skip rows if needed
    std::string line;
    int64_t current_row = 0;
    while (current_row < start_row && std::getline(file, line)) {
        current_row++;
        
        if (current_row % 50000 == 0) {
            std::cout << "Skipped " << current_row << " rows" << std::endl;
        }
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
        
        if (row % 10000 == 0) {
            std::cout << "Loaded " << row << " rows from CSV (starting from row " << start_row << ")" << std::endl;
        }
    }
    
    std::cout << "Loaded " << row << " rows from CSV (starting from row " << start_row << ")" << std::endl;
    return host_data;
}

void ivf_flat_build_cluster_segment_assignment_global(raft::device_resources const& dev_resources,
                                                      const cuvs::neighbors::ivf_flat::index_params& index_params,
                                                      raft::device_matrix_view<const half, int64_t> dataset)
{
  auto stream = raft::resource::get_cuda_stream(dev_resources);

  using namespace cuvs::neighbors;

  std::cout << "Building IVF-Flat index: Cluster-segment Assignment-global" << std::endl;
  
  auto start = std::chrono::high_resolution_clock::now();
  
  auto index = ivf_flat::build(dev_resources, index_params, dataset);
  raft::resource::sync_stream(dev_resources, stream);
  int64_t n_rows = dataset.extent(0);
  raft::device_vector<uint32_t, int64_t> new_labels = raft::make_device_mdarray<uint32_t>(
    dev_resources, raft::resource::get_large_workspace_resource(dev_resources), raft::make_extents<int64_t>(n_rows)); 

  ivf_flat::compute_labels(dev_resources, &index, dataset, new_labels, n_rows);
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Build cluster segment assignment global time: " << elapsed.count() << " ms" << std::endl;

  // print_device_matrix(dev_resources, new_labels.view());
}

 // Start of Selection
void ivf_flat_build_cluster_segment_assignment_local(raft::device_resources const& dev_resources,
                                                     const cuvs::neighbors::ivf_flat::index_params& index_params,
                                                     raft::device_matrix_view<const half, int64_t> dataset)
{
  auto stream = raft::resource::get_cuda_stream(dev_resources);
  
  using namespace cuvs::neighbors;

  std::cout << "Building IVF-Flat index: Cluster-segment Assignment-local" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  
  auto index = ivf_flat::build(dev_resources, index_params, dataset);
  raft::resource::sync_stream(dev_resources, stream);
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Build cluster segment assignment local time: " << elapsed.count() << " ms" << std::endl;

  raft::device_vector_view<uint32_t, int64_t> new_labels = index.train_labels();

  // print_device_matrix(dev_resources, new_labels);
}

void ivf_flat_build_cluster_segment_assignment_local_stream(raft::device_resources const& dev_resources,
                                                            const cuvs::neighbors::ivf_flat::index_params& index_params,
                                                            std::vector<raft::device_matrix_view<const half, int64_t>> &dataset)
{
  int num_streams = dataset.size();
  // Create a CUDA stream pool
  raft::resource::set_cuda_stream_pool(dev_resources, std::make_shared<rmm::cuda_stream_pool>(num_streams));
  
  // Vector to store threads
  std::vector<std::thread> threads;
  threads.reserve(num_streams);
  
  // Launch a thread for each dataset
  for (int i = 0; i < num_streams; i++) {
    threads.emplace_back([&dev_resources, &index_params, &dataset, i]() {
      // Get a stream from the pool
      auto stream = raft::resource::get_next_usable_stream(dev_resources, i);
      // Create a device_resources object with the stream
      raft::device_resources stream_pool_handle(dev_resources);
      raft::resource::set_cuda_stream(stream_pool_handle, stream);
      
      using namespace cuvs::neighbors;
      
      // Build index
      auto index = ivf_flat::build(stream_pool_handle, index_params, dataset[i]);
      raft::device_vector_view<uint32_t, int64_t> new_labels = index.train_labels();
    });
  }
  
  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }
}

// returns a gpu pointer. 
cuvs::neighbors::ivf_flat::index<half, int64_t>* get_index(raft::device_resources const& dev_resources)
{
  return new cuvs::neighbors::ivf_flat::index<half, int64_t>(dev_resources);
}

void build_global(raft::device_resources const& dev_resources, 
                  cuvs::neighbors::ivf_flat::index<half,int64_t>& idx)
{
  
}
void ivf_flat_build_global(raft::device_resources const& dev_resources,
                           const cuvs::neighbors::ivf_flat::index_params& index_params,
                           raft::device_matrix_view<const half, int64_t> dataset)
{
  using namespace cuvs::neighbors;

  std::cout << "Building IVF-Flat index: Global" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  
  auto index = ivf_flat::build(dev_resources, index_params, dataset);
  cudaStreamSynchronize(0);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Build global time: " << elapsed.count() << " ms" << std::endl;

  raft::device_vector_view<uint32_t, int64_t> new_labels = index.train_labels();
  // print_device_matrix(dev_resources, new_labels);
}


void ivf_flat_build_global_stream(raft::device_resources const& dev_resources,
                                 const cuvs::neighbors::ivf_flat::index_params& index_params,
                                 std::vector<raft::device_matrix_view<const half, int64_t>> &dataset)
{
  int num_streams = dataset.size();
  // Create a CUDA stream pool
  raft::resource::set_cuda_stream_pool(dev_resources, std::make_shared<rmm::cuda_stream_pool>(num_streams));
  
  // Vector to store threads
  std::vector<std::thread> threads;
  threads.reserve(num_streams);
  
  // Launch a thread for each dataset
  for (int i = 0; i < num_streams; i++) {
    threads.emplace_back([&dev_resources, &index_params, &dataset, i]() {
      // Get a stream from the pool
      auto stream = raft::resource::get_next_usable_stream(dev_resources, i);
      // Create a device_resources object with the stream
      raft::device_resources stream_pool_handle(dev_resources);
      raft::resource::set_cuda_stream(stream_pool_handle, stream);
      
      using namespace cuvs::neighbors;
      
      // Build index
      auto index = ivf_flat::build(stream_pool_handle, index_params, dataset[i]);
    });
  }
  
  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }
}


int main()
{
  raft::device_resources dev_resources;
  
  // Set pool memory resource with 12 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 12ull * 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);
  
  
  auto index = get_index(dev_resources); // 需要确认这个 128 是真的在最后 build 的时候被使用
  std::cout << "n_lists = " << index->list_sizes().extent(0) << std::endl;  
  // Define dataset dimensions
  // int64_t n_dim = 128;
  // int64_t n_samples_per_head = 130293; // Number of vectors per head
  // int num_heads = 8; // Total number of heads in the CSV file

  // // CSV file has shape (1042344, 128), which is approximately 8 heads of 130293 vectors each
  // std::string csv_path = "/home/v-xle/cuvs/examples/cpp/src/csv-data/key_states_layer_10.csv";
  
  // // Create vectors to hold all datasets
  // std::vector<raft::device_matrix<float, int64_t>> fp32_datasets;
  // std::vector<raft::device_matrix<__half, int64_t>> fp16_datasets;
  // std::vector<raft::device_matrix_view<const half, int64_t>> dataset_views;
  
  // auto stream = raft::resource::get_cuda_stream(dev_resources);
  
  // // Load data for each head
  // for (int head = 0; head < num_heads; head++) {
  //   std::cout << "Loading dataset for head " << head << " from CSV..." << std::endl;
    
  //   // Load dataset from CSV
  //   int64_t start_row = head * n_samples_per_head;
  //   auto host_dataset_fp32 = load_csv(dev_resources, csv_path, start_row, n_samples_per_head, n_dim);
    
  //   // Create device matrices
  //   fp32_datasets.push_back(raft::make_device_matrix<float, int64_t>(dev_resources, n_samples_per_head, n_dim));
  //   fp16_datasets.push_back(raft::make_device_matrix<__half, int64_t>(dev_resources, n_samples_per_head, n_dim));
    
  //   // Copy host data to device
  //   raft::copy(fp32_datasets[head].data_handle(), host_dataset_fp32.data_handle(), n_samples_per_head * n_dim, stream);
    
  //   // Synchronize to ensure data is copied before proceeding
  //   raft::resource::sync_stream(dev_resources, stream);
  //   std::cout << "Data for head " << head << " copied to device successfully" << std::endl;
    
  //   // Convert FP32 to FP16
  //   size_t total_elements = n_samples_per_head * n_dim;
  //   int threadsPerBlock = 256;
  //   int blocksDataset = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    
  //   std::cout << "Converting dataset for head " << head << " from FP32 to FP16..." << std::endl;
  //   convert_float_to_half<<<blocksDataset, threadsPerBlock>>>(
  //     fp32_datasets[head].data_handle(), fp16_datasets[head].data_handle(), total_elements);
    
  //   raft::resource::sync_stream(dev_resources, stream);
    
  //   // Add the dataset view to our vector
  //   dataset_views.push_back(raft::make_const_mdspan(fp16_datasets[head].view()));
  // }
  
  // std::cout << "All datasets loaded and converted to FP16" << std::endl;

  
  // // Create index parameters
  // cuvs::neighbors::ivf_flat::index_params global_params;
  // global_params.n_lists = 4096;
  // global_params.kmeans_trainset_fraction = 1;
  // global_params.metric = cuvs::distance::DistanceType::InnerProduct;
  // global_params.add_data_on_build = false;
  
  // cuvs::neighbors::ivf_flat::index_params segment_params = global_params;
  // segment_params.segment_build = true;
  // segment_params.segment_count = 64;
  // segment_params.kmeans_n_iters = 20;
  
  
  // Test multi-stream assign local
  // std::cout << "Testing ivf_flat_build_cluster_segment_assignment_local_stream with " << num_heads << " datasets..." << std::endl;
  // auto start = std::chrono::high_resolution_clock::now();
  
  // ivf_flat_build_cluster_segment_assignment_local_stream(dev_resources, segment_params, dataset_views);
  // raft::resource::sync_stream(dev_resources);
  
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> elapsed = end - start;
  // std::cout << "Multi-stream build time for " << num_heads << " heads: " << elapsed.count() << " ms" << std::endl;
  
  // // For comparison, run the single-stream versions on all datasets sequentially
  // std::cout << "\nFor comparison, running single-stream builds on all " << num_heads << " datasets sequentially..." << std::endl;
  // start = std::chrono::high_resolution_clock::now();
  
  // for (int i = 0; i < num_heads; i++) {
  //   std::cout << "Processing dataset " << i << " with single-stream..." << std::endl;
  //   ivf_flat_build_cluster_segment_assignment_local(dev_resources,
  //                                                 segment_params,
  //                                                 dataset_views[i]);
  // }
  // end = std::chrono::high_resolution_clock::now();
  // elapsed = end - start;
  // std::cout << "Sequential single-stream build time for all " << num_heads << " datasets: " << elapsed.count() << " ms" << std::endl;

  // Test multi-stream global build
  
  // std::cout << "\nTesting ivf_flat_build_global_stream with " << num_heads << " datasets..." << std::endl;
  // auto start = std::chrono::high_resolution_clock::now();
    
  // ivf_flat_build_global_stream(dev_resources, global_params, dataset_views);
  // raft::resource::sync_stream(dev_resources);
  
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> elapsed = end - start;
  // std::cout << "Multi-stream global build time for " << num_heads << " heads: " << elapsed.count() << " ms" << std::endl;
  
  // // For comparison, run the sequential global builds
  // std::cout << "\nFor comparison, running single-stream global builds on all " << num_heads << " datasets sequentially..." << std::endl;
  // start = std::chrono::high_resolution_clock::now();
  
  // for (int i = 0; i < num_heads; i++) {
  //   std::cout << "Processing dataset " << i << " with single-stream global build..." << std::endl;
  //   ivf_flat_build_global(dev_resources, global_params, dataset_views[i]);
  // }
  // end = std::chrono::high_resolution_clock::now();
  // elapsed = end - start;
  // std::cout << "Sequential single-stream global build time for all " << num_heads << " datasets: " << elapsed.count() << " ms" << std::endl;
}