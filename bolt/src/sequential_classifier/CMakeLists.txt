add_subdirectory(tests)

add_library(sequential_classifier_lib PipelineBuilder.cc SchemaProcessor.cc)
target_link_libraries(
  sequential_classifier_lib
  PRIVATE
  bolt_network_lib
  bolt_datasets_lib
  hash_function_lib
)