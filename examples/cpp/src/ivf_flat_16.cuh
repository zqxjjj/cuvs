#ifndef IVF_FLAT_16_CUH
#define IVF_FLAT_16_CUH

// Include standard headers first
#include <chrono>
#include <iostream>
#include <cstdint>
#include <optional>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Include project-specific headers
#include "common.cuh"
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

// CUDA streams and resources
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>

// Thrust
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

namespace cuvs_utils {

/**
 * @brief Print device matrix content to console
 */
inline void print_device_matrix(const raft::device_resources& handle,
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
inline raft::host_matrix<float, int64_t> load_csv(const raft::resources& handle, 
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
inline void build_segment_global(raft::device_resources const& dev_resources,
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
inline void build_segment_global_multistream(raft::device_resources const& dev_resources,
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
inline cuvs::neighbors::ivf_flat::index<half, int64_t>* get_index(raft::device_resources const& dev_resources)
{
  cuvs::neighbors::ivf_flat::index_params params = cuvs::neighbors::ivf_flat::index_params();
  return new cuvs::neighbors::ivf_flat::index<half, int64_t>(dev_resources, params, 128);
}

/**
 * @brief Build local segment index
 */
inline void build_segment_local(raft::device_resources const& dev_resources,
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
  raft::copy(dataset_keys.data_handle(), keys_view_host.data_handle(), seq_len * 128, stream); // TODO: try to eliminate the copy
  raft::resource::sync_stream(dev_resources, stream);
  
  auto build_params = ivf_flat::index_params();
  build_params.n_lists = n_clusters;
  build_params.segment_build = true;
  
  ivf_flat::build(dev_resources, build_params, raft::make_const_mdspan(dataset_keys.view()), index);
}

/**
 * @brief Build local segment index with multiple streams for parallelization
 */
inline void build_segment_local_multistream(raft::device_resources const& dev_resources,
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
inline void build_global(raft::device_resources const& dev_resources, 
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
inline void build_global_multistream(raft::device_resources const& dev_resources,
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

} // namespace cuvs_utils

// Function declaration for build_test
int build_test(torch::Tensor& input_keys);

#endif // IVF_FLAT_16_CUH