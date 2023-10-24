# -*- coding: utf-8 -*-
import multiprocessing
import os
import re
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

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
    is_public_release = "THIRDAI_EXPOSE_ALL" not in feature_flags
else:
    feature_flags = "THIRDAI_BUILD_LICENSE THIRDAI_CHECK_LICENSE"
    is_public_release = True


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

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            # not used on MSVC, but no harm
            "-DCMAKE_BUILD_TYPE={}".format(build_mode),
        ]

        # To build the wheel, we don't need to make "all" targets. We just need
        # what is required to be packaged with python.  The pybind11 target in
        # CMakeLists.txt is defined as _thirdai, which is what is shipped with
        # the built wheel. Since setup.py is for use in packaging the wheel, we
        # pass this specific target to cmake args to avoid having to compile
        # tests and other potential libraries/executables created.
        #
        # Equivalent to calling `make _thirdai`.
        build_args = ["-t", "_thirdai"]

        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # Detect if user wants to use ccache from a CMake variable
        # If set to 0 (also used as a default when unset) ccache is disabled.
        # Otherwise ccache is enabled.
        use_ccache = os.environ.get("USE_CCACHE", "0")
        if use_ccache != "0":
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            ]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401

                    cmake_args += ["-GNinja"]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(
                        build_mode.upper(), extdir
                    )
                ]
                build_args += ["--config", build_mode]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        build_args += ["-j{}".format(num_jobs)]
        cmake_args += [f"-DTHIRDAI_FEATURE_FLAGS={feature_flags}"]

        build_dir = "build/"
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_dir)


version = None
with open("thirdai.version") as version_file:
    version = version_file.read().strip()
    suffix = os.environ.get("THIRDAI_BUILD_IDENTIFIER", None)
    if suffix:
        version = "{}+{}".format(version, suffix)

neural_db_deps = [
    "PyTrie",
    "PyMuPDF==1.22.5",
    "langchain",
    "bs4",
    "trafilatura",
    "python-docx",
    "url-normalize",
    "nltk",
    "unidecode",
    "pydantic",
    "unstructured[all-docs]",
    "sortedcontainers",
    "SQLAlchemy",
]

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="thirdai",
    version=version,
    author="ThirdAI",
    author_email="contact@thirdai.com",
    description="A faster cpu machine learning library",
    long_description="""
      A faster cpu machine learning library that uses sparsity and hashing to 
      accelerate inference and training. See https://thirdai.com for more 
      details.
    """,
    license_files=("LICENSE.txt",),
    ext_modules=[CMakeExtension("thirdai._thirdai")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=[
        "numpy",
        "typing_extensions",
        "requests",
        "pandas>=2.0.0",
    ],
    extras_require={
        # The cryptography requirement is necessary to avoid ssl errors
        # The tokenizers requirement ensures that all of the [test] depedencies are
        # installable from a wheel on an m1.
        # The latest version of MLFLOW available while building wheels is 1.23.1
        # For MLFLOW to work, the alembic version of the backend databases for local
        # MLFLOW and server MLFLOW should be the same. Hence, we are fixing the
        # version of MLFLOW here. The version of protobuf that works with this
        # MLFLOW is also being fixed.
        "neural_db": neural_db_deps,
        "test": [
            "pytest",
            "pytest-mock",
            "boto3",
            "moto[server]",
            "mlflow",
            "protobuf",
            "datasets",
            "torch",
            "toml",
            "psutil",
            "transformers",
            "pandas>=2.0.0",
            "cryptography",
            "tokenizers",
            "pyarrow",
            "prometheus_client",
            "gcsfs",
            "s3fs",
            "mock",
            "scikit-learn",
            "beir",
            "pyOpenSSL>22.1.0",
            "ray",
        ]
        + neural_db_deps,
        "benchmark": [
            "toml",
            "psutil",
            "scikit-learn",
            "mlflow",
            "protobuf",
            "boto3",
            "s3fs",
            "gcsfs",
            "nltk",
            "ray[default]",
            "torch",
        ]
        + neural_db_deps,
        # See https://github.com/readthedocs/sphinx_rtd_theme/issues/1343 for why we restrict the sphinx version
        "docs": ["sphinx!=5.2.0.post0", "sphinx_rtd_theme"],
    },
    packages=["thirdai"]
    + [
        "thirdai." + p
        for p in find_packages(
            where="thirdai_python_package",
            # We don't want the experimental submodule included in releases.
            exclude=["experimental"] if is_public_release else [],
        )
    ],
    license="proprietary",
    package_dir={"thirdai": "thirdai_python_package"},
)
