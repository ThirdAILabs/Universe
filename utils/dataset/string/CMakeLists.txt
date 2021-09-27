enable_testing()

#add_library(string_dataset_lib STATIC StringDataset.cc)
add_library(string_dataset_lib STATIC GlobalFreq.cc)

#target_link_libraries(string_dataset_lib PRIVATE vectorizer_lib)

add_subdirectory(loaders)
add_subdirectory(vectorizers)