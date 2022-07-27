add_subdirectory(tests)

add_library(sequential_classifier_lib SequentialClassifierPipelineBuilder.cc)
target_link_libraries(
  sequential_classifier_lib
  PRIVATE
  bolt_network_lib
  auto_classifier_lib
  hash_function_lib
)