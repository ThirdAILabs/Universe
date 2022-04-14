add_library(license_lib STATIC DecodeLicense.cc EncodeLicense.cc)

target_link_libraries(
  license_lib
  PRIVATE
  libcryptopp
)
