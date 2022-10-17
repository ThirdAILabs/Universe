

add_library(flash_indexer_lib STATIC Indexer.cc Flash.cc)

target_link_libraries(flash_indexer_lib PRIVATE hash_function_lib hash_table_lib license_wrapper_lib)

target_link_libraries(flash_indexer_lib PUBLIC OpenMP::OpenMP_CXX)

