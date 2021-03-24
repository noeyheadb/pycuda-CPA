# pycuda-CPA
CUDA implementation of CPA(Correlation Power Analysis) using [`pycuda`](https://github.com/inducer/pycuda). It can be used as a sub-package in your Python project by adding it as a submodule.
1. Move to the directory where the pycudaCPA package will be added.
2. `git submodule add https://github.com/noeyheadb/pycuda-CPA.git pycudaCPA`.

## Environment setup
- GPUs with compute capability less than 1.3 are not supported.
- Make `pycuda` available by installing the CUDA toolkit and setting environment variables, etc.
- Test code of `pycuda` is [here](https://documen.tician.de/pycuda/).

## Requirements
- pycuda >= 2020.1
- numpy
