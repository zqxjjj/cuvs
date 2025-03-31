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

#include <cuda_fp16.h>

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

void ivf_flat_build_cluster_segment_assignment_global(raft::device_resources const& dev_resources,
                                                      const cuvs::neighbors::ivf_flat::index_params& index_params,
                                                      raft::device_matrix_view<const half, int64_t> dataset,
                                                      raft::device_matrix_view<const half, int64_t> queries)
{
  using namespace cuvs::neighbors;

  std::cout << "Building IVF-Flat index: Cluster-segment Assignment-local" << std::endl;
  
  auto start = std::chrono::high_resolution_clock::now();
  
  auto index = ivf_flat::build(dev_resources, index_params, dataset);
  int64_t n_rows = dataset.extent(0);
  raft::device_vector<uint32_t, int64_t> new_labels = raft::make_device_mdarray<uint32_t>(
    dev_resources, raft::resource::get_large_workspace_resource(dev_resources), raft::make_extents<int64_t>(n_rows)); 

  ivf_flat::compute_labels(dev_resources, &index, dataset, new_labels, n_rows);
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Build cluster segment assignment global time: " << elapsed.count() << " ms" << std::endl;

  print_device_matrix(dev_resources, new_labels.view());
}

void ivf_flat_build_cluster_segment_assignment_local(raft::device_resources const& dev_resources,
                                                      const cuvs::neighbors::ivf_flat::index_params& index_params,
                                                      raft::device_matrix_view<const half, int64_t> dataset,
                                                      raft::device_matrix_view<const half, int64_t> queries)
{
  using namespace cuvs::neighbors;

  std::cout << "Building IVF-Flat index: Cluster-segment Assignment-local" << std::endl;
  auto index = ivf_flat::build(dev_resources, index_params, dataset);

  raft::device_vector_view<uint32_t, int64_t> new_labels = index.train_labels();

  print_device_matrix(dev_resources, new_labels);
}

void ivf_flat_build_global(raft::device_resources const& dev_resources,
                           const cuvs::neighbors::ivf_flat::index_params& index_params,
                           raft::device_matrix_view<const half, int64_t> dataset,
                           raft::device_matrix_view<const half, int64_t> queries)
{
  using namespace cuvs::neighbors;

  std::cout << "Building IVF-Flat index: Global" << std::endl;
  
  auto index = ivf_flat::build(dev_resources, index_params, dataset);

  raft::device_vector_view<uint32_t, int64_t> new_labels = index.train_labels();

  print_device_matrix(dev_resources, new_labels);
}

int main()
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Create input arrays.
  int64_t n_samples = 10000;
  int64_t n_dim     = 3;
  int64_t n_queries = 10;
  auto dataset_fp32      = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries_fp32      = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);

  generate_dataset(dev_resources, dataset_fp32.view(), queries_fp32.view());

  auto dataset_fp16 = raft::make_device_matrix<__half, int64_t>(dev_resources, n_samples, n_dim);
  auto queries_fp16 = raft::make_device_matrix<__half, int64_t>(dev_resources, n_queries, n_dim);

  size_t total_dataset_elements = n_samples * n_dim;
  size_t total_queries_elements = n_queries * n_dim;

  int threadsPerBlock = 256;
  int blocksDataset = (total_dataset_elements + threadsPerBlock - 1) / threadsPerBlock;
  int blocksQueries = (total_queries_elements + threadsPerBlock - 1) / threadsPerBlock;

  convert_float_to_half<<<blocksDataset, threadsPerBlock>>>(dataset_fp32.data_handle(), dataset_fp16.data_handle(), total_dataset_elements);
  convert_float_to_half<<<blocksQueries, threadsPerBlock>>>(queries_fp32.data_handle(), queries_fp16.data_handle(), total_queries_elements);

  cudaStreamSynchronize(0);
  
  // Create index parameters
  cuvs::neighbors::ivf_flat::index_params global_params;
  global_params.n_lists = 4096;
  global_params.kmeans_trainset_fraction = 1;
  global_params.metric = cuvs::distance::DistanceType::InnerProduct;
  global_params.add_data_on_build = false;
  
  cuvs::neighbors::ivf_flat::index_params segment_params = global_params;
  segment_params.segment_build = true;
  segment_params.segment_count = 64;
  segment_params.kmeans_n_iters = 10;
  
  // Simple build and search example.
  ivf_flat_build_global(dev_resources,
                        global_params,
                        raft::make_const_mdspan(dataset_fp16.view()),
                        raft::make_const_mdspan(queries_fp16.view()));
  ivf_flat_build_cluster_segment_assignment_global(dev_resources,
                                                   segment_params,
                                                   raft::make_const_mdspan(dataset_fp16.view()),
                                                   raft::make_const_mdspan(queries_fp16.view()));
  ivf_flat_build_cluster_segment_assignment_local(dev_resources,
                                                   segment_params,
                                                   raft::make_const_mdspan(dataset_fp16.view()),
                                                   raft::make_const_mdspan(queries_fp16.view()));

}
