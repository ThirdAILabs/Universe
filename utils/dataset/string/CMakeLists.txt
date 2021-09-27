enable_testing()

add_library(string_dataset_lib STATIC StringDataset.cc)
#add_library(string_dataset_lib STATIC GlobalFreq.cc)

#target_link_libraries(string_dataset_lib PRIVATE vectorizer_lib)
target_link_libraries(string_dataset_lib PRIVATE tri_gram_vectorizer_lib sentence_loader_lib)

add_subdirectory(loaders)
add_subdirectory(vectorizers)