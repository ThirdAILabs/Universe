add_library(auto_classifier_lib AutoClassifierUtils.cc)

target_link_libraries(auto_classifier_lib bolt_network_lib)

add_subdirectory(tests)

enable_testing()