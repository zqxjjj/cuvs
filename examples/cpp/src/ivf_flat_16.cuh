#ifndef IVF_FLAT_16_CUH
#define IVF_FLAT_16_CUH

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
int build_test(torch::Tensor& input_keys);

#endif // IVF_FLAT_16_CUH