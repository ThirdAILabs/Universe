configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/Version.h @ONLY)

add_library(logging_utils_lib Logging.cc)
target_link_libraries(logging_utils_lib PUBLIC spdlog)
