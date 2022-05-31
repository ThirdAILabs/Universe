enable_testing()

add_executable(
    bloom_filter_test
    BloomFilterTest.cc
)

target_link_libraries(
    bloom_filter_test
    PRIVATE
    gtest_main
    utils_lib
    hash_function_lib
)

include(GoogleTest)
gtest_discover_tests(bloom_filter_test)