enable_testing()

add_library(vectorizer_lib STATIC TriGramVectorizer.cc UnigramVectorizer.cc)

target_link_libraries(vectorizer_lib PRIVATE hash_function_lib)