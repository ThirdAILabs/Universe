# -*- coding: utf-8 -*-
import os
import re
import subprocess
import sys
import multiprocessing

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Default is release build with full parallelism
if "THIRDAI_NUM_JOBS" in os.environ:
    num_jobs = os.environ["THIRDAI_NUM_JOBS"]
else:
    num_jobs = multiprocessing.cpu_count() * 2
if "THIRDAI_BUILD_MODE" in os.environ:
    build_mode = os.environ["THIRDAI_BUILD_MODE"]
else:
    build_mode = "Release"
if "THIRDAI_FEATURE_FLAGS" in os.environ:
    feature_flags = os.environ["THIRDAI_FEATURE_FLAGS"]
else:
    feature_flags = "THIRDAI_BUILD_LICENSE;THIRDAI_CHECK_LICENSE"


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        global build_mode
        global feature_flags
        global num_jobs

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        build_dir = "build/"

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DCMAKE_BUILD_TYPE={}".format(build_mode),
        ]
        build_args = []

        build_args += [f"-j{num_jobs}"]
        cmake_args += [f'"-DFEATURE_FLAGS={feature_flags}"']

        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        cmake_call = f"cmake {ext.sourcedir} {' '.join(cmake_args)}"
        subprocess.check_call(cmake_call, cwd=build_dir, shell=True)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_dir)


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="thirdai",
    version="0.1.2",
    author="ThirdAI",
    author_email="contact@thirdai.com",
    description="A faster cpu machine learning library",
    long_description="""
      A faster cpu machine learning library that uses sparsity and hashing to 
      accelerate inference and training. See https://thirdai.com for more 
      details.
    """,
    ext_modules=[CMakeExtension("thirdai._thirdai")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    extras_require={
        "test": ["pytest"],
    },
    packages=["thirdai"]
    + ["thirdai." + p for p in find_packages(where="thirdai_python_package")],
    licence="proprietary",
    package_dir={"thirdai": "thirdai_python_package"},
)
