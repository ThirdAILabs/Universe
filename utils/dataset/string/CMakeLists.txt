enable_testing()

add_library(string_dataset_lib STATIC StringDataset.cc GlobalFreq.cc)
# add_library(string_dataset_lib STATIC GlobalFreq.cc)

target_link_libraries(string_dataset_lib PRIVATE string_vectorizer_lib string_loader_lib)

add_subdirectory(loaders)
add_subdirectory(vectorizers)