add_library(bolt_network_lib FullyConnectedNetwork.cc DistributedModel.cc)

target_link_libraries(bolt_network_lib PUBLIC bolt_layer_lib license_wrapper_lib OpenMP::OpenMP_CXX)

add_subdirectory(tests)

enable_testing()
