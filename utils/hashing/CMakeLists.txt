add_library(dwta_lib STATIC DWTA.cc)

add_library(hash_function_lib STATIC UniversalHash.cc MurmurHash.cc FastSRP.cc SRP.cc DensifiedMinHash.cc)
