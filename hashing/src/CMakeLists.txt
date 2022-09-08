add_library(hash_function_lib STATIC UniversalHash.cc DWTA.cc MurmurHash.cc FastSRP.cc SRP.cc DensifiedMinHash.cc)
target_link_libraries(hash_function_lib PRIVATE OpenMP::OpenMP_CXX)
