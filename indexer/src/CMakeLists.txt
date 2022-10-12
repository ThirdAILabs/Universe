
add_library(
    flash_indexer_lib SHARAED 
    Indexer.h
    Flash.h
    Flash.cc
)

target_link_libraries(
  flash_indexer_lib 
  PRIVATE 
  hash_function_lib
  hash_table_lib
  license_wrapper_lib
)