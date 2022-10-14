add_executable(create_signed_license CreateSignedLicense.cc)
target_link_libraries(create_signed_license PRIVATE thirdai ${CRYPTOPP_LIB})
