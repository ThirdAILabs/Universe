add_library(bolt_network_lib FullyConnectedNetwork.cc DLRM.cc)

target_link_libraries(bolt_network_lib bolt_layer_lib bolt_loss_function_lib)

add_subdirectory(tests)

enable_testing()