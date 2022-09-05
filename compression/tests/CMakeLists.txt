enable_testing()

add_executable(dragon_vector_test DragonVectorTest.cc)
target_link_libraries(
	dragon_vector_test
	PRIVATE
	compression_core_lib
	gtest_main
)

add_executable(count_sketch_test CountSketchTest.cc)
target_link_libraries(
	count_sketch_test 
	PRIVATE 
	compression_core_lib 
	gtest_main
)

include(GoogleTest)

set(COMPRESSION_TESTS
	dragon_vector_test
	count_sketch_test
)

foreach(COMPRESSION_TEST IN LISTS COMPRESSION_TESTS)
	gtest_discover_tests(${COMPRESSION_TEST} DISCOVERY_TIMEOUT ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()