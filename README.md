# Universe 
Main Repository (follow monorepositry) 

[Repository Setup Instructions](https://www.notion.so/Universe-Setup-ed71176c2cf44b038ece8aee9fb64d35)

## Development Scripts
There are some script in the bin folder that allow you to easily build, test,
and lint you code.
1. Run `$ bin/build.py` from anywhere to have cmake build everything in universe.
All executables will be in their corresponding directory within the build 
directory. e.g. the mpi_example executable will be in 
`Universe/build/examples/mpi-example/mpi_example`. By default this will
run in parallel and build all unbuilt targets or targets whose component source
files have been updated. You can pass in parameters to run in serial and
build in other build modes (Debug, etc.). Run bin/build.py -h for more info. 
This will also install the thirdai package using pip.
2. Run `$ bin/cpp-test.sh` after running `bin/build.py` from anywhere to have 
cmake run all c++ tests. To run specific tests, you can also pass a regular 
expression to filter tests (or provide an explicit test name):
`$ bin/cpp-test.sh -R <test filter expression>`.
Note you can actually pass any arguments you would pass to ctest to this
script and it will forward them. 
3. Run `$ bin/python-test.sh` from anywhere to run all python tests. To run specific
tests, you can also pass a regular expression to filter tests 
(or provide an explicit test name):
`$ bin/python-test.sh -R <test filter expression>`.
You can also filter to unit, integration, or benchmarking tests by e.g. running 
`$ bin/python-test.sh -m unit`.
Note you can actually pass any arguments you would pass to pytest to this
script and it will forward them, see https://docs.pytest.org/en/6.2.x/usage.html.
You can also directly run tests using pytest, but this script also ensures
that the thirdai so file is on your path. 
3. Run `$ bin/cpp-format.sh` from anywhere to format all C++ code.
4. Run `$ bin/python-format.sh` from anywhere to format all Python code.
4. Run `$ bin/lint.sh` from anywhere to run linting on all code (primarily 
these are clang-tidy checks).
5. Run `$ bin/generate_compile_commands.sh` from anywhere to generate the compile
commands database (you shouldn't often need to do this manually, but try it
if your intellisense is acting strangely).

## Docker
1. You can run all of the above scripts, including building and testing your
code, in our development docker container. To do this, you can create the 
development docker container by running `$ bin/docker-build.sh` and enter
a bash shell in the container by running `$ bin/docker-run.sh`. Universe will
be mounted as a folder in the root of the container. Note that there may be 
some issues with existing cache information if you have last build on your local
machine, and you may need to run a clean build or delete your build folder.

## Feature Flags
See https://www.notion.so/Feature-Flags-dbff0b88985242b8a07cc58a9261a1e2 for
information about creating and using feature flags, and 
https://www.notion.so/Feature-Flag-List-e006f6be1abe4d46bde97cf63c97a0f5
for a current list of feature flags and what they do. Please update the list
whenever you add a new feature flag.

## Debugging your Pybound C++ code from python
1. Simply build the code in RelWithDebInfo or Debug mode, and
run your python code with perf or gdb. To run with gdb, run `gdb python3` and then
`your_py_script.py` (and feel free to set breakpoints in C++ code). 
2. You can also now run performance profile your python code. To run with
perf, simply run normal perf commands attached to your python process. 
2. Debugging Pybound code with ASan isn't supported anymore, if you really need
it we can add it back.

## Debugging your C++ code
1. The simplest way it to build your code in RelWithDebInfo or Debug mode and 
then run gdb.
2. If you want to run with ASan (an adress sanitizier), build with RelWithAsan
or DebugWithAsan.

## Manual building and testing (DEPRECATED, use scripts in bin, see above)
1. Clone this repository and navigate into it.
2. `$ mkdir build && cd build`
3. `$ cmake ..` - this will generate the makefile that can be used to compile the code. 
4. `$ make all` will compile all targets. Or you can use `$ make <target name>` to build a specific target. Run with -j$NUM_JOBS
to build in parallel ($NUM_JOBS should be ~1.5 times the total number of threads on your machine
for best performance).
5. All executables will be in their corresponding directory within the build directory. e.g. the mpi_example executable will be in `Universe/build/examples/mpi-example/mpi_example`.
6. To run tests simply run `$ ctest` from within the build directory after compiling all of the targets, or optionally pass a regular expression to filter tests (or provide an explicit test names) `$ ctest -R <test filter expression>`. 

## Installing python bindings
1. Running `bin/build.py` as normal will install python bindings on your machine
automatically using pip.
2. To build a wheel, run `python3 setup.py bdist_wheel`. This will generate a
.whl file in the dist folder specific to your architecture and operating system.
Note that this will do a release build with release flags; to build a wheel
with other flags, you need to set environment variables (see setup.py for more
details). In fact, `bin/build.py` sets environment variables in its process 
namespace and then calls setup.py internally. You can then install 
the wheel by running `pip3 install dist/<wheel>`. You can also upload whl files
to pypi, but this should only be done with a release build, ideally as part of
a CI pipeline.

## Using cmake
To understand how to setup a executable, library, or test using cmake please see the examples in the `examples` directory. For more context here are a few things to know: 
1. `add_library` - this creates a library of the specified name using the given targets. The `SHARED` or `STATIC` identifier means that the library will be either shared and dynamically loaded or static and linked in to other binaries that depend on it. For our purposes we should probably just use `STATIC`. 
2. `target_link_libraries` - this links a specified library to another target, either a library or an executable.
3. `add_executable` - creates an executable from the given files. 
4. `add_subdirectory` - cmake starts at the CMakeLists.txt in the root of the project and traverses down, this command basically tells cmake that it must look in the given subdirectory to find more build targets. 
5. `enable_testing` - this allows cmake to find the tests in the build targets. Every intermediate directory between the root directory and the directory containing the tests must have this in its CMakeLists.txt in order for cmake to find the tests. 
6. `include(GoogleTest)` and `gtest_discover_tests` - this imports googletest and tells it to find all of the tests in the specified target for cmake. For an example please see `examples/test`.
7. `find_package(MPI)` and `target_link_libraries(<some target> PRIVATE MPI::MPI_CXX)` this finds the mpi library and links it with the given target. For an example please see `examples/mpi-example`.