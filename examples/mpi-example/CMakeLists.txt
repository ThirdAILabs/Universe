set(CMAKE_CXX_COMPILER "mpicxx")
set(CMAKE_C_COMPILER "mpicc")

add_executable(mpi_example mpi_hello_world.cpp)
