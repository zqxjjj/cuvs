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

# 显式添加Python头文件路径
set(PYTHON_INCLUDE_DIR "/usr/include/python3.9")
include_directories(${PYTHON_INCLUDE_DIR})
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-I${PYTHON_INCLUDE_DIR}>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-I${PYTHON_INCLUDE_DIR}>")

# 配置 PyTorch
find_package(Torch REQUIRED PATHS ${CMAKE_PREFIX_PATH} 
             HINTS ${CMAKE_PREFIX_PATH} 
             PATHS `python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`)
message(STATUS "Found PyTorch: ${TORCH_INCLUDE_DIRS}")

set(TORCH_INCLUDE_DIRS
${TORCH_INSTALL_PREFIX}/include
${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include)
message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")

include_directories(${TORCH_INCLUDE_DIRS})

# # Find PyTorch package for Python bindings
# find_package(Torch REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
# message(STATUS "Found PyTorch: ${TORCH_INCLUDE_DIRS}")

# # Find pybind11 for Python bindings
# find_package(pybind11 QUIET)
# if(NOT pybind11_FOUND)
#   include(FetchContent)
#   FetchContent_Declare(
#     pybind11
#     GIT_REPOSITORY https://github.com/pybind/pybind11.git
#     GIT_TAG        v2.13.6  # Adjust version as needed
#   )
#   FetchContent_MakeAvailable(pybind11)
# endif()
# message(STATUS "Found pybind11: ${pybind11_INCLUDE_DIRS}")
# # Add PyTorch and pybind11 include directories globally
# include_directories(${TORCH_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

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
