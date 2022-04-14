add_library(license_lib STATIC DecodeLicense.cc EncodeLicense.cc)

message(CRYPTO ${CRYPTOPP_LIB})

target_link_libraries(
  license_lib
  PRIVATE
  ${CRYPTOPP_LIB}
)
