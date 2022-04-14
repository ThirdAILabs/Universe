enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

add_executable(encode_license src/EncodeLicense.cc)
add_executable(decode_license src/DecodeLicense.cc)

target_link_libraries(encode_license PRIVATE ${CRYPTOPP_LIB})
target_link_libraries(decode_license PRIVATE ${CRYPTOPP_LIB})