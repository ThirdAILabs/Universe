enable_testing()

add_library(string_vectorizer_lib STATIC TriGramVectorizer.cc UnigramVectorizer.cc)

target_link_libraries(string_vectorizer_lib PRIVATE hash_function_lib)
