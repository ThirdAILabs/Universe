add_library(auto_classifier_lib AutoClassifierBase.cc)

target_link_libraries(auto_classifier_lib bolt_graph_lib)

add_subdirectory(tests)

enable_testing()