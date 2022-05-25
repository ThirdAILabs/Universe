add_library(bolt_text_classifier_lib TextClassifier.cc)
target_link_libraries(
  bolt_text_classifier_lib 
  PRIVATE
  bolt_network_lib
  bolt_datasets_lib
  hash_function_lib
)