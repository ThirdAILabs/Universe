add_library(bolt_tabular_classifier_lib TabularClassifier.cc)
target_link_libraries(
  bolt_tabular_classifier_lib 
  PRIVATE
  bolt_network_lib
  bolt_datasets_lib
  hash_function_lib
)