enable_testing()

add_library(dataset_core_lib BlockBatchProcessor.cc)
target_link_libraries(dataset_core_lib PUBLIC OpenMP::OpenMP_CXX)
