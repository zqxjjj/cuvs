from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Standard compilation flags from CMakeLists.txt
cuda_compile_args = [
    '-O3',
    '-std=c++17',
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE',  # Add this line
]


# CXX compilation flags
cxx_compile_args = [
    '-O3',
    '-std=c++17',
]

# Include directories as specified
include_dirs = [
    '/usr/include',
    "/usr/include/x86_64-linux-gnu",
    '/home/v-xle/miniconda3/envs/cuvs/include',
    '/home/v-xle/cuvs/examples/cpp/libtorch/include',
    # '/home/v-xle/miniconda3/pkgs/libraft-headers-only-25.04.00a49-cuda12_250330_ge8c05b79_49/include',
    # '/home/v-xle/miniconda3/envs/cuvs/include',
    # '/home/v-xle/miniconda3/pkgs/librmm-25.4.0a51-cuda12_250330_c6773f27/include/rapids',
    # '/home/v-xle/miniconda3/pkgs/cuda-cudart-dev_linux-64-12.8.90-h3f2d84a_1/targets/x86_64-linux/include',
]
# Link arguments aligned with CMakeLists.txt
link_args = [
    '-lcuda', 
    '-lcudart', 
    '-lcublas', 
    '-lcusolver', 
    '-lcusparse', 
    '-lcurand',
    '-lrmm',
]

ivf_flat_module = CUDAExtension(
    'ivf_flat_perf',
    sources=['ivf_flat_fp16_example.cu'],
    extra_compile_args={
        'cxx': cxx_compile_args, 
        'nvcc': cuda_compile_args
    },
    extra_link_args=link_args,
    include_dirs=include_dirs,
    language='cuda'
)

setup(
    name='ivf_flat_perf',
    ext_modules=[ivf_flat_module],
    cmdclass={"build_ext": BuildExtension},
    install_requires=['pybind11', 'torch'],
    python_requires='>=3.6',
)
