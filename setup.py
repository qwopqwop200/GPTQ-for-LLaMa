from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='gptq_llama',
    version='0.1',
    description='GPTQ for Llama',
    package_dir={'': 'src'},
    packages=['gptq_llama', 'gptq_llama.quant_cuda'],
    ext_modules=[CUDAExtension(
        'gptq_llama.quant_cuda', ['src/gptq_llama/quant_cuda/quant_cuda.cpp', 'src/gptq_llama/quant_cuda/quant_cuda_kernel.cu']
    )],
    install_requires=['torch'],
    cmdclass={'build_ext': BuildExtension}
)
