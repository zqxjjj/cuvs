#include "ivf_flat_16.cuh"

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

// test build_segment_local and build_segment_local_multistream
int build_test(torch::Tensor& input_keys, int num_segments, int n_clusters)
{
  using namespace cuvs::neighbors;
  raft::device_resources dev_resources;
  // Get dimensions from input tensor
  int kv_head_num = input_keys.size(0);
  int seq_len = input_keys.size(1);
  int dim = input_keys.size(2);

  std::cout << "Input tensor dimensions: [" << kv_head_num << ", " << seq_len << ", " << dim << "]" << std::endl;
  
  // compute storage size for one index (one index for one head)  
  std::size_t quantizer_bytes = n_clusters * 128 * 4; // to store centroids,  n_clusters * dim * sizeof(float)
  std::size_t input_keys_bytes = seq_len * 128 * 2; // to store input keys
  std::size_t inds_bytes = seq_len * 4; // to store inds_ptrs_; identifier for each vector
  std::size_t labels_bytes = seq_len * 4; // to store train_labels_; identifier for each vector
  std::size_t total_memory_bytes = quantizer_bytes + input_keys_bytes + inds_bytes + labels_bytes;
  std::cout << "Total memory bytes: " << total_memory_bytes << " bytes" << std::endl;
  std::size_t overhead_factor = 3;
  std::size_t pool_size = total_memory_bytes * overhead_factor;
  // Round up to the nearest multiple of 256 bytes
  pool_size = (pool_size + 255) & ~(std::size_t)255;
  std::cout << "Pool size: " << pool_size << " bytes" << std::endl;
  std::cout << "dev_resources allocates " << static_cast<float>(pool_size) / 1024 / 1024 / 1024 << " GB" << std::endl;

  // Set pool memory resource with 12 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), pool_size);
  rmm::mr::set_current_device_resource(&pool_mr);
  
  // Print tensor dimensions  
  auto stream = raft::resource::get_cuda_stream(dev_resources);
  
  // Calculate pointers for each head
  std::vector<uint16_t*> keys_ptrs(kv_head_num);
  uint16_t* keys_base_ptr = reinterpret_cast<uint16_t*>(input_keys.data_ptr<at::Half>());
  for (int head = 0; head < kv_head_num; head++) {
    // Each head's data starts at head * seq_len * dim elements from the base pointer
    keys_ptrs[head] = keys_base_ptr + (head * seq_len * dim);
    std::cout << "Head " << head + 1 << " of " << kv_head_num << " ready" << std::endl;
  }
  
  // ---------- Serial processing with build_segment_local ----------
  std::cout << "\nTesting build_segment_local (serial processing):" << std::endl;
  
  // Create indices for serial processing
  std::vector<cuvs::neighbors::ivf_flat::index<half, int64_t>*> serial_indices(kv_head_num);
  for (int i = 0; i < kv_head_num; i++) {
    serial_indices[i] = cuvs_utils::get_index(dev_resources);
  }
  
  // Start timing for serial processing
  auto serial_start = std::chrono::high_resolution_clock::now();
  
  // Process each head serially
  for (int head = 0; head < kv_head_num; head++) {
    cuvs_utils::build_segment_local(dev_resources, *serial_indices[head], 
                        keys_ptrs[head], seq_len, n_clusters, num_segments); // need to support count_segment. 
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
  std::vector<cuvs::neighbors::ivf_flat::index<half, int64_t>*> parallel_indices(kv_head_num);
  for (int i = 0; i < kv_head_num; i++) {
    parallel_indices[i] = cuvs_utils::get_index(dev_resources);
  }
  
  // Start timing for parallel processing
  auto parallel_start = std::chrono::high_resolution_clock::now();
  
  // Process all heads in parallel
  cuvs_utils::build_segment_local_multistream(dev_resources, parallel_indices, keys_ptrs, seq_len, n_clusters, num_segments);
  
  // End timing for parallel processing
  auto parallel_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> parallel_elapsed = parallel_end - parallel_start;
  
  std::cout << "Parallel processing completed successfully" << std::endl;
  std::cout << "Total parallel build time: " << parallel_elapsed.count() << " ms" << std::endl;
  
  // Speedup calculation
  double speedup = serial_elapsed.count() / parallel_elapsed.count();
  std::cout << "\nSpeedup from parallelization: " << speedup << "x" << std::endl;
  
  // Clean up resources
  for (int i = 0; i < kv_head_num; i++) {
    ivf_flat::helpers::reset_index(dev_resources, serial_indices[i]);
    delete serial_indices[i];
    
    ivf_flat::helpers::reset_index(dev_resources, parallel_indices[i]);
    delete parallel_indices[i];
  }
  return 0;
}