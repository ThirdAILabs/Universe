enable_testing()

add_executable(
    global_freq_test
    GlobalFreqTest.cc
)

add_executable(
    uni_gram_vectorizer_test
    UnigramVectorizerTest.cc
)

# add_executable(
#     tri_gram_vectorizer_test
#     TriGramVectorizerTest.cc
# )

add_executable(
    sentence_loader_test
    SentenceLoaderTest.cc
)

add_executable(
    string_dataset_test
    StringDatasetTest.cc
)

target_link_libraries(
    global_freq_test
    PRIVATE
    gtest_main
    string_dataset_lib
    # string_loader_lib
)

target_link_libraries(
    uni_gram_vectorizer_test
    PRIVATE
    string_vectorizer_lib
    string_loader_lib
    string_dataset_lib
    gtest_main
)

# target_link_libraries(
#     tri_gram_vectorizer_test
#     PRIVATE
#     string_vectorizer_lib
#     gtest_main
# )

target_link_libraries(
    sentence_loader_test
    PRIVATE
    string_loader_lib
    gtest_main
)

target_link_libraries(
    string_dataset_test
    PRIVATE
    string_dataset_lib
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(uni_gram_vectorizer_test)
# gtest_discover_tests(tri_gram_vectorizer_test)
gtest_discover_tests(sentence_loader_test)
gtest_discover_tests(string_dataset_test)
gtest_discover_tests(global_freq_test)