add_library(flash_lib STATIC Flash.cc)

target_link_libraries(
  flash_lib 
  PRIVATE 
  hash_function_lib
  hash_table_lib
  dataset_lib
)
