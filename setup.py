# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import glob
import os
import numpy
import platform
from setuptools import setup
from setuptools import find_packages

import torch
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension


class CustomedBuildExt(BuildExtension):
  """Avoid a gcc warning below:
    
    cc1plus: warning: command line option '-Wstrict-prototypes' is valid
    for C/ObjC but not for C++
  """

  def build_extensions(self):
    if '-Wstrict-prototypes' in self.compiler.compiler_so:
      self.compiler.compiler_so.remove('-Wstrict-prototypes')
    self.compiler.compiler_so.append('-fopenmp')
    super().build_extensions()


def BuildPyTorchKernel():
  """building pytorch kernel
  """
  # disable compile on MacOS
  if 'Darwin' in platform.platform() or 'macOS' in platform.platform():
    return []
  define_macros = []
  extra_compile_args = {"cxx": []}
  extension = CppExtension

  # compile - default to cxx
  build_dir = f'{os.path.dirname(os.path.abspath(__file__))}/csrc/'
  source = glob.glob(build_dir + "/**/*.cc", recursive=True)
  source += glob.glob(build_dir + "/**/*.cpp", recursive=True)

  # add includes
  build_dirs = [build_dir, ]
  for sub in os.listdir(build_dir):
    sub_dir = os.path.join(build_dir, sub)
    if os.path.isdir(sub_dir):
      build_dirs.append(sub_dir)

  # build with cuda
  if torch.cuda.is_available():
    extension = CUDAExtension
    source += glob.glob(build_dir + "/**/*.cu", recursive=True)
    extra_compile_args.update({
        "nvcc": [
            # "-O2",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            # selective_scan
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            # "--ptxas-options=-v",
            # "-lineinfo",
        ],
        "cxx": [
            "-g",
            "-fopenmp",
            "-std=c++17",
            "-O3",
        ],
    })

  ext_modules = [
      extension(
          "mamba",
          source,
          include_dirs=build_dirs,
          extra_compile_args=extra_compile_args,
      )
  ]
  return ext_modules

setup(
    name="mamba",
    author="Kai Jin",
    author_email="atranitell@gmail.com",
    long_description_content_type='text/markdown',
    license="Apache 2.0 Licence",
    packages=find_packages(),
    ext_modules=BuildPyTorchKernel(),
    cmdclass={'build_ext': CustomedBuildExt},
    classifiers=[
        'Programming Language :: Python :: 3.10',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
