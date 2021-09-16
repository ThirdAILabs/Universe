find_package(MPI REQUIRED)

add_library(distributed_sparse_layer_lib DistributedSparseLayer.cc)

target_link_libraries(
  distributed_sparse_layer_lib 
  PRIVATE 
  bolt_lib
  MPI::MPI_CXX
)