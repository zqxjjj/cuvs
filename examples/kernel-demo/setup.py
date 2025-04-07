from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


my_module = CUDAExtension(
    'my_kernel',
    sources=['kernel.cu'],
    extra_compile_args={'cxx': ['-O3', '-std=c++17'], 
                        'nvcc': ['-O3', '-std=c++17', '--expt-relaxed-constexpr']},
    extra_link_args=['-lcuda', '-lcudart'],
    language='cuda'
)

setup(
    name='my_kernel',
    ext_modules=[my_module],
    cmdclass={"build_ext": BuildExtension},
    install_requires=['pybind11', 'torch'],
    python_requires='>=3.10',
)
