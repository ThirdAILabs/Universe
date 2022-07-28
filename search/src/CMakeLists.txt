
add_library(maxflash_lib STATIC MaxFlash.cc MaxFlashArray.cc)

target_link_libraries(
  maxflash_lib 
  PUBLIC 
  hash_function_lib
  hash_table_lib
  license_wrapper_lib
  OpenMP::OpenMP_CXX
)
