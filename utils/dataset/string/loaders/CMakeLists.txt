enable_testing()

add_library(string_loader_lib STATIC SentenceLoader.h)

target_link_libraries(string_loader_lib PRIVATE string_dataset_lib)