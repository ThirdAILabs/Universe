add_library(flash_lib STATIC Flash.cc)

add_library(maxflash_lib STATIC MaxFlash.cc MaxFlashArray.cc)

target_link_libraries(
  flash_lib 
  PRIVATE 
  hash_function_lib
  hash_table_lib
  license_wrapper_lib
)

target_link_libraries(
  maxflash_lib 
  PRIVATE 
  hash_function_lib
  hash_table_lib
  license_wrapper_lib
)
