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

// Include standard headers first
#include <chrono>
#include <iostream>

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

// Include our utility header last
#include "ivf_flat_fp16_utils.cuh"

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

// test build_segment_local and build_segment_local_multistream
int build_test(torch::Tensor& input_keys)
{
  using namespace cuvs::neighbors;
  raft::device_resources dev_resources;
  
  // Set pool memory resource with 12 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 12ull * 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);
  
  // Get dimensions from input tensor
  int kv_head_num = input_keys.size(0);
  int seq_len = input_keys.size(1);
  int dim = input_keys.size(2);
  
  // Fixed parameters
  int n_clusters = 4096; // Number of clusters for IVF-Flat
  int n_heads = 8; // Total number of heads to process (use the first 8 heads from input tensor)
  
  // Print tensor dimensions
  std::cout << "Input tensor dimensions: [" << kv_head_num << ", " << seq_len << ", " << dim << "]" << std::endl;
  std::cout << "Using first " << n_heads << " heads from tensor" << std::endl;
  
  auto stream = raft::resource::get_cuda_stream(dev_resources);
  
  // Vectors to store per-head data
  std::vector<uint16_t*> keys_ptrs(n_heads);
  std::vector<int> seq_lengths(n_heads, seq_len);
  std::vector<int> n_clusters_list(n_heads, n_clusters);
  
  // Get pointer to the input tensor data
  uint16_t* keys_base_ptr = reinterpret_cast<uint16_t*>(input_keys.data_ptr<at::Half>());
  
  // Calculate pointers for each head
  for (int head = 0; head < n_heads; head++) {
    // Each head's data starts at head * seq_len * dim elements from the base pointer
    keys_ptrs[head] = keys_base_ptr + (head * seq_len * dim);
    std::cout << "Head " << head + 1 << " of " << n_heads << " ready" << std::endl;
  }
  
  // ---------- Serial processing with build_segment_local ----------
  std::cout << "\nTesting build_segment_local (serial processing):" << std::endl;
  
  // Create indices for serial processing
  std::vector<cuvs::neighbors::ivf_flat::index<half, int64_t>*> serial_indices(n_heads);
  for (int i = 0; i < n_heads; i++) {
    serial_indices[i] = cuvs_utils::get_index(dev_resources);
  }
  
  // Start timing for serial processing
  auto serial_start = std::chrono::high_resolution_clock::now();
  
  // Process each head serially
  for (int head = 0; head < n_heads; head++) {
    cuvs_utils::build_segment_local(dev_resources, *serial_indices[head], 
                        keys_ptrs[head], seq_len, n_clusters);
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
    parallel_indices[i] = cuvs_utils::get_index(dev_resources);
  }
  
  // Start timing for parallel processing
  auto parallel_start = std::chrono::high_resolution_clock::now();
  
  // Process all heads in parallel
  cuvs_utils::build_segment_local_multistream(dev_resources, parallel_indices, keys_ptrs, seq_lengths, n_clusters_list);
  
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_test", &build_test, "Test IVF-Flat index building (CUDA)");
}