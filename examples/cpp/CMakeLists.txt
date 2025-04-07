# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(../cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(test_cuvs)

project(test_cuvs LANGUAGES CXX CUDA)
find_package(Threads)

# 设置CUDA_HOME环境变量，帮助PyTorch找到CUDA
if(NOT DEFINED ENV{CUDA_HOME} AND NOT DEFINED ENV{CUDA_PATH})
  get_filename_component(CUDA_HOME ${CMAKE_CUDA_COMPILER} DIRECTORY)
  get_filename_component(CUDA_HOME ${CUDA_HOME} DIRECTORY)
  set(ENV{CUDA_HOME} ${CUDA_HOME})
  message(STATUS "Setting CUDA_HOME to: ${CUDA_HOME}")
endif()

# 确保CUDA路径也被添加到CMAKE_PREFIX_PATH
if(DEFINED ENV{CUDA_HOME})
  list(APPEND CMAKE_PREFIX_PATH $ENV{CUDA_HOME})
  message(STATUS "Added CUDA_HOME to CMAKE_PREFIX_PATH: $ENV{CUDA_HOME}")
endif()

# 找到Python解释器和开发头
find_package(Python3 REQUIRED COMPONENTS Interpreter Development) 

# 使用正确的Python包含路径
include_directories(${Python3_INCLUDE_DIRS})
# 打印Python包含路径用于调试
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")

# 配置 PyTorch
# PyTorch 默认下载的是 CPU 版本的, 但是好像是 GPU 版本的才可以成功. https://pytorch.org/get-started/locally/
execute_process(
  COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "PyTorch CMake prefix path: ${TORCH_PREFIX_PATH}")

find_package(Torch REQUIRED PATHS ${TORCH_PREFIX_PATH})
message(STATUS "Found PyTorch: ${TORCH_INCLUDE_DIRS}")

set(TORCH_INCLUDE_DIRS
${TORCH_INSTALL_PREFIX}/include
${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include)
message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")
# ------------- configure cuvs -----------------#

rapids_cpm_init()
set(BUILD_CUVS_C_LIBRARY OFF)
include(../cmake/thirdparty/get_cuvs.cmake)

# -------------- compile tasks ----------------- #
add_executable(BRUTE_FORCE_EXAMPLE src/brute_force_bitmap.cu)
add_executable(CAGRA_EXAMPLE src/cagra_example.cu)
add_executable(CAGRA_PERSISTENT_EXAMPLE src/cagra_persistent_example.cu)
add_executable(DYNAMIC_BATCHING_EXAMPLE src/dynamic_batching_example.cu)
add_executable(IVF_FLAT_EXAMPLE src/ivf_flat_example.cu)
add_executable(IVF_FLAT_16 src/ivf_flat_16.cu)
add_executable(IVF_PQ_EXAMPLE src/ivf_pq_example.cu)
add_executable(VAMANA_EXAMPLE src/vamana_example.cu)

# Install headers for external use
install(FILES src/ivf_flat_fp16_utils.cuh
        DESTINATION include/cuvs/examples/cpp)

# `$<TARGET_NAME_IF_EXISTS:conda_env>` is a generator expression that ensures that targets are
# installed in a conda environment, if one exists
target_link_libraries(BRUTE_FORCE_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(CAGRA_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(
  CAGRA_PERSISTENT_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env> Threads::Threads
)
target_link_libraries(
  DYNAMIC_BATCHING_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env> Threads::Threads
)
target_link_libraries(IVF_PQ_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(IVF_FLAT_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(IVF_FLAT_16 PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(VAMANA_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)

# 配置 pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6  # Adjust version as needed
  )
  FetchContent_MakeAvailable(pybind11)
endif()
message(STATUS "Found pybind11: ${pybind11_INCLUDE_DIRS}")
# Add PyTorch and pybind11 include directories globally
include_directories(${TORCH_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

pybind11_add_module(ivf_flat_16 MODULE src/ivf_flat_16.cu)