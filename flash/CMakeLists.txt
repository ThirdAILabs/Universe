enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

add_executable(run_flash_experimental experimental/RunFlash.cc)
target_link_libraries(run_flash_experimental PRIVATE flash_lib)