add_subdirectory(batch_types)

add_library(dataset_lib Dataset.cc)
target_link_libraries(dataset_lib batch_lib)
