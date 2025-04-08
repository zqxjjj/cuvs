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

# 添加RPATH设置以确保conda环境中的库被正确找到
if(DEFINED ENV{CONDA_PREFIX})
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  if(APPLE)
    set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib;@loader_path")
  else()
    set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib;$ORIGIN")
  endif()
  message(STATUS "Setting RPATH to conda environment: $ENV{CONDA_PREFIX}/lib")
  
  # 添加conda环境的include目录
  include_directories($ENV{CONDA_PREFIX}/include)
  message(STATUS "Adding conda include directory: $ENV{CONDA_PREFIX}/include")
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


# Apply PyTorch compiler flags to ensure consistent ABI
# 为了避免符号未定义等链接错误，一定要使用 PyTorch 提供的编译标志。find_package(Torch) 会提供变量 TORCH_CXX_FLAGS，应将其加入编译选项。
message(STATUS "PyTorch CXX Flags: ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CXX_FLAGS}")

set(TORCH_INCLUDE_DIRS
${TORCH_INSTALL_PREFIX}/include
${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include)
message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")

# Make sure the correct path to PyTorch libraries is used
message(STATUS "PyTorch Libraries: ${TORCH_LIBRARIES}")
# Add PyTorch library directory to library search path
link_directories(${TORCH_INSTALL_PREFIX}/lib)

# ------------- configure cuvs -----------------#

rapids_cpm_init()
set(BUILD_CUVS_C_LIBRARY OFF)
include(../cmake/thirdparty/get_cuvs.cmake)

# -------------- compile tasks ----------------- #
# Install headers for external use
install(FILES src/ivf_flat_fp16_utils.cuh
        DESTINATION include/cuvs/examples/cpp)

# add_executable(IVF_FLAT_16 src/ivf_flat_16.cu)
add_library(IVF_FLAT_16 SHARED src/ivf_flat_16.cu) # create .so file , 加这个 SHARED 解决了 multiple definition 的问题
# `$<TARGET_NAME_IF_EXISTS:conda_env>` is a generator expression that ensures that targets are
# installed in a conda environment, if one exists
target_link_libraries(IVF_FLAT_16 PRIVATE cuvs::cuvs raft::raft $<TARGET_NAME_IF_EXISTS:conda_env> ${TORCH_LIBRARIES})
set_target_properties(IVF_FLAT_16 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(IVF_FLAT_16 PRIVATE LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

# 显式添加RAFT头文件路径
target_include_directories(IVF_FLAT_16 PRIVATE /home/v-xle/miniconda3/envs/cuvs/include)

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

pybind11_add_module(ivf_flat_16p MODULE src/ivf_flat_16_binder.cu)
target_link_libraries(ivf_flat_16p PRIVATE IVF_FLAT_16)
# Set RPATH for the Python module to find PyTorch libraries
target_compile_definitions(ivf_flat_16p PRIVATE LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)