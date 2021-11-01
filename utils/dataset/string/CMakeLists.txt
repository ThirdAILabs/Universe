enable_testing()


add_library(string_dataset_lib STATIC StringFactory.h)

target_link_libraries(string_dataset_lib PRIVATE string_vectorizer_lib string_loader_lib string_global_frequency_lib)


add_subdirectory(loaders)
add_subdirectory(vectorizers)
add_subdirectory(global_frequency)
