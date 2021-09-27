enable_testing()

add_library(string_dataset_lib STATIC StringDataset.cc)

target_link_libraries(string_dataset_lib PRIVATE tri_gram_vectorizer_lib sentence_loader_lib)

add_subdirectory(loaders)
add_subdirectory(vectorizers)