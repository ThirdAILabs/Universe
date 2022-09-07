add_library(hash_table_lib STATIC VectorHashTable.cc SampledHashTable.cc)
target_link_libraries(hash_table_lib PUBLIC OpenMP::OpenMP_CXX)
