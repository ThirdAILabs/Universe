enable_testing()

add_executable(
    tri_gram_vectorizer_test
    TriGramVectorizerTest.cc
)

target_link_libraries(
    tri_gram_vectorizer_test
    PRIVATE
    tri_gram_vectorizer_lib
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(tri_gram_vectorizer_test)