add_library(bolt_network_lib FullyConnectedNetwork.cc DLRM.cc DistributedModel.cc Model.cc )

target_link_libraries(bolt_network_lib bolt_layer_lib  license_wrapper_lib)

add_subdirectory(tests)

enable_testing()