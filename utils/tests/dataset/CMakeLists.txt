enable_testing()

add_subdirectory(svm)
# add_executable(
#     tri_gram_vectorizer_test
#     TriGramVectorizerTest.cc
# )
add_executable(
    uni_gram_vectorizer_test
    UnigramVectorizerTest.cc
)

# target_link_libraries(
#     tri_gram_vectorizer_test
#     PRIVATE
#     tri_gram_vectorizer_lib
#     gtest_main
# )
target_link_libraries(
    uni_gram_vectorizer_test
    PRIVATE
    vectorizer_lib
    gtest_main
)

include(GoogleTest)
# gtest_discover_tests(tri_gram_vectorizer_test)
gtest_discover_tests(uni_gram_vectorizer_test)
add_subdirectory(string)
