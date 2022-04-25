
add_library(license_wrapper_lib LicenseWrapper.cc)

if (${LICENSE_BUILD_FLAG_FOUND})
target_link_libraries(license_wrapper_lib PRIVATE ${CRYPTOPP_LIB})
endif()

