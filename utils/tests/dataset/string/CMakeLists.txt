enable_testing()

add_executable(
    tri_gram_vectorizer_test
    TriGramVectorizerTest.cc
)

add_executable(
    sentence_loader_test
    SentenceLoaderTest.cc
)

target_link_libraries(
    tri_gram_vectorizer_test
    PRIVATE
    tri_gram_vectorizer_lib
    gtest_main
)

target_link_libraries(
    sentence_loader_test
    PRIVATE
    sentence_loader_lib
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(tri_gram_vectorizer_test)
gtest_discover_tests(sentence_loader_test)