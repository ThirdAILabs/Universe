enable_testing()

add_executable(
    word_uni_gram_vectorizer_test
    WordUniGramVectorizerTest.cc
)

add_executable(
    char_tri_gram_vectorizer_test
    CharTriGramVectorizerTest.cc
)

# add_executable(
#     sentence_loader_test
#     SentenceLoaderTest.cc
# )

# add_executable(
#     string_dataset_test
#     StringDatasetTest.cc
# )

target_link_libraries(
    word_uni_gram_vectorizer_test
    PRIVATE
    string_vectorizer_lib
    gtest_main
)

target_link_libraries(
    char_tri_gram_vectorizer_test
    PRIVATE
    string_vectorizer_lib
    gtest_main
)

# target_link_libraries(
#     sentence_loader_test
#     PRIVATE
#     string_loader_lib
#     gtest_main
# )

# target_link_libraries(
#     string_dataset_test
#     PRIVATE
#     gtest_main
# )

include(GoogleTest)
gtest_discover_tests(word_uni_gram_vectorizer_test)
gtest_discover_tests(char_tri_gram_vectorizer_test)
# gtest_discover_tests(sentence_loader_test)
# gtest_discover_tests(string_dataset_test)