enable_testing()

add_executable(dragon_vector_test DragonVectorTest.cc)

target_link_libraries(
	dragon_vector_test
	PRIVATE
	compression_core_lib
	gtest_main
)

add_executable(default_compressed_vector_test DefaultCompressedVectorTest.cc)

target_link_libraries(
	default_compressed_vector_test
	PRIVATE
	compression_core_lib
	gtest_main
)

include(GoogleTest)
gtest_discover_tests(dragon_vector_test)
gtest_discover_tests(default_compressed_vector_test)