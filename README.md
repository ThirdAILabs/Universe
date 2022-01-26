# Universe 
Main Repository (follow monorepositry) 

[Repository Setup Instructions](https://docs.google.com/document/d/196ajXaVUUqpTFigkMmBhdlhTAiKy7LxgYS95xqpB2Ys/edit?usp=sharing)

## Development Scripts
There are some script in the bin folder that allow you to easily build, test,
and lint you code.
1. Run `$ bin/build.sh` from anywhere to have cmake build everything in universe.
All executables will be in their corresponding directory within the build 
directory. e.g. the mpi_example executable will be in 
`Universe/build/examples/mpi-example/mpi_example`.
2. Run `$ bin/tests.sh` from anywhere to have cmake run all tests. To run specific
tests, you can also pass a regular expression to filter tests 
(or provide an explicit test name):
`$ bin/tests.sh -R <test filter expression>`
Note you can actually pass any arguments you would pass to ctest to this
script and it will forward them. 
3. Run `$ bin/format.sh` from anywhere to format all code.
4. Run `$ bin/lint.sh` from anywhere to run linting on all code (primarily 
these are clang-tidy checks).
5. Run `$ generate_compile_commands.sh` from anywhere to generate the compile
commands database (you shouldn't ever need to call this script manually).


## Development (Deprecated, use scripts in bin, see above)
1. Clone this repository and navigate into it.
2. `$ mkdir build && cd build`
3. `$ cmake ..` - this will generate the makefile that can be used to compile the code. 
4. `$ make all` will compile all targets. Or you can use `$ make <target name>` to build a specific target. 
5. All executables will be in their corresponding directory within the build directory. e.g. the mpi_example executable will be in `Universe/build/examples/mpi-example/mpi_example`.
6. To run tests simply run `$ ctest` from within the build directory after compiling all of the targets, or optionally pass a regular expression to filter tests (or provide an explicit test names) `$ ctest -R <test filter expression>`. 

## Installing python bindings
1. The building target `thirdai` will compile the `thirdai.so` library in the build directory. 
This is automatically run on a full build, so you can run `bin/build.sh` as normal.
2. To compile and install bindings, run `pip3 install .` from the root of the Universe directory. 
This will allow them to be imported anywhere on your machine,

## Using cmake
To understand how to setup a executable, library, or test using cmake please see the examples in the `examples` directory. For more context here are a few things to know: 
1. `add_library` - this creates a library of the specified name using the given targets. The `SHARED` or `STATIC` identifier means that the library will be either shared and dynamically loaded or static and linked in to other binaries that depend on it. For our purposes we should probably just use `STATIC`. 
2. `target_link_libraries` - this links a specified library to another target, either a library or an executable.
3. `add_executable` - creates an executable from the given files. 
4. `add_subdirectory` - cmake starts at the CMakeLists.txt in the root of the project and traverses down, this command basically tells cmake that it must look in the given subdirectory to find more build targets. 
5. `enable_testing` - this allows cmake to find the tests in the build targets. Every intermediate directory between the root directory and the directory containing the tests must have this in its CMakeLists.txt in order for cmake to find the tests. 
6. `include(GoogleTest)` and `gtest_discover_tests` - this imports googletest and tells it to find all of the tests in the specified target for cmake. For an example please see `examples/test`.
7. `find_package(MPI)` and `target_link_libraries(<some target> PRIVATE MPI::MPI_CXX)` this finds the mpi library and links it with the given target. For an example please see `examples/mpi-example`.