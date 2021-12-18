add_library(bolt_network_lib Network.cc)

target_link_libraries(bolt_network_lib bolt_layer_lib)

add_library(dlrm_lib DLRM.cc)
target_link_libraries(dlrm_lib bolt_layer_lib)
