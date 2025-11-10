#!/usr/bin/env bash

# CuPy on AMD GPU has experimental support
# Need to build CuPy from source to run on AMD GPU
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/opt/rocm
export HCC_AMDGPU_TARGET=gfx1010

# AMD software stack that includes programming models, 
# tools, compilers, libraries, and runtimes for AMD GPUs
source /opt/rocm/share/rocprofiler-register/setup-env.sh
