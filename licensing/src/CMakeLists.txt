add_executable(create_signed_license file/CreateSignedLicense.cc)
target_link_libraries(create_signed_license PRIVATE cereal::cereal
                                                    cryptopp::cryptopp)
