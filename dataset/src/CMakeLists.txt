enable_testing()

add_executable(dcd DynamicCountsDiagnostics.cc)
target_link_libraries(dcd PRIVATE hash_function_lib)

add_subdirectory(string)
add_subdirectory(core)
add_subdirectory(bolt_datasets)
