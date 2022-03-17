# Universe 
Main Repository (follow monorepositry) 

[Repository Setup Instructions](https://docs.google.com/document/d/196ajXaVUUqpTFigkMmBhdlhTAiKy7LxgYS95xqpB2Ys/edit?usp=sharing)

## Development Scripts
There are some script in the bin folder that allow you to easily build, test,
and lint you code.
1. Run `$ bin/build.sh` from anywhere to have cmake build everything in universe.
All executables will be in their corresponding directory within the build 
directory. e.g. the mpi_example executable will be in 
`Universe/build/examples/mpi-example/mpi_example`. By default this will
run in parallel and build all unbuilt targets or targets whose component source
files have been updated, but you can pass in parameters to run in serial or build
only a specific target. See the source of `$ bin/build.sh` for more details.
2. Run `$ bin/cpp-test.sh` from anywhere to have cmake run all c++ tests. To run specific
tests, you can also pass a regular expression to filter tests 
(or provide an explicit test name):
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

## Debugging with GDB and ASan from Python
1. Do a debug build. If you try to run a python test at this point, you may
encounter errors or the python test won't run at all.
2. Run `export LD_PRELOAD=$(gcc -print-file-name=libasan.so):$LD_PRELOAD`. This
is NOT run automatically when you startup a development docker container, as 
it makes all python code run twice as slow and it should be a conscious
decision. If you try to run a python test now, you will get lot's of noise
memory leak errors from python itself. In fact, even running python3 and 
immediately exiting will cause errors.
3. Run `export LSAN_OPTIONS=suppressions=~/Universe/leak-suppresions.supp`. This will
supress memory leak errors from Python calls (basically this is all memory 
leaks, so ASan will mostly be useful for illegal memory accesses). This DOES
get run automatically as part of the docker container.
4. Run your python code/test with perf or gdb. You can also just run it normally
and ASan will run.
See https://github.com/google/sanitizers/issues/1086 and 
https://github.com/tobywf/python-ext-asan for more details.


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
1. The building target `thirdai` will compile the `thirdai.so` library in the build directory. 
This is automatically run on a full build, so you can run `bin/build.sh` as normal.
Note this will use the version of python you get from running `which python3`, 
and even with the PYTHONPATH changes below a different version of python will
not be able to find the so file.
2. To use the bindings, you need to tell python where the `thirdai.so` file is.
To do this, you can add the so file to your PYTHONPATH environment
variable. This will allow you to automatically use the newest built version of 
the library. To do this, you can run
`export PYTHONPATH=~/Universe/build:$PYTHONPATH`
(replace the path with the correct one if your Universe folder is not in your
home directory). This will work until you open a new shell; to 
automatically update your PYTHONPATH when you start your shell add the above
command to your ~/.bash_profile or ~/.bash_rc, or equivalently run
`echo "export PYTHONPATH=~/Universe/build:\$PYTHONPATH" >> $HOME/.bash_profile`. 
Alternatively you can run `pip3 install .`. This installs thirdi without messing
around with environment variables, but is not preferred for development since it
is performs an entirely seperate parallel build from `bin/build.sh`, and so is
much slower.

## Using cmake
To understand how to setup a executable, library, or test using cmake please see the examples in the `examples` directory. For more context here are a few things to know: 
1. `add_library` - this creates a library of the specified name using the given targets. The `SHARED` or `STATIC` identifier means that the library will be either shared and dynamically loaded or static and linked in to other binaries that depend on it. For our purposes we should probably just use `STATIC`. 
2. `target_link_libraries` - this links a specified library to another target, either a library or an executable.
3. `add_executable` - creates an executable from the given files. 
4. `add_subdirectory` - cmake starts at the CMakeLists.txt in the root of the project and traverses down, this command basically tells cmake that it must look in the given subdirectory to find more build targets. 
5. `enable_testing` - this allows cmake to find the tests in the build targets. Every intermediate directory between the root directory and the directory containing the tests must have this in its CMakeLists.txt in order for cmake to find the tests. 
6. `include(GoogleTest)` and `gtest_discover_tests` - this imports googletest and tells it to find all of the tests in the specified target for cmake. For an example please see `examples/test`.
7. `find_package(MPI)` and `target_link_libraries(<some target> PRIVATE MPI::MPI_CXX)` this finds the mpi library and links it with the given target. For an example please see `examples/mpi-example`.