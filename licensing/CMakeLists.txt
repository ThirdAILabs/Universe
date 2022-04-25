enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

add_executable(create_signed_license src/CreateSignedLicense.cc)
target_link_libraries(create_signed_license PRIVATE ${CRYPTOPP_LIB})
