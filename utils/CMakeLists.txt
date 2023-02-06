enable_testing()

add_subdirectory(tests)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/Version.h @ONLY)
