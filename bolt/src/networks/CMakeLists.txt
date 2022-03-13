add_library(bolt_network_lib Model.cc FullyConnectedNetwork.cc DLRM.cc)

target_link_libraries(bolt_network_lib bolt_layer_lib)

add_subdirectory(tests)

enable_testing()