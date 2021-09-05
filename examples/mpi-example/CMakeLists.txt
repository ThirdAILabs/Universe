find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(mpi_example mpi_hello_world.cpp)

target_link_libraries(mpi_example ${MPI_CXX_LIBRARIES})