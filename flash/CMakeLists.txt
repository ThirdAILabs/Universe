enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

add_executable(run_flash RunFlash.cc)
target_link_libraries(run_flash PRIVATE flash_lib svm_dataset_lib)