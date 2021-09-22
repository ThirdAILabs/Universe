file(GLOB SOURCES src/*.cc)

add_library(bolt_lib STATIC ${SOURCES})
target_link_libraries(bolt_lib PRIVATE dwta_lib sampled_hash_table_lib)

add_executable(bolt bolt.cc)
target_link_libraries(bolt PRIVATE bolt_lib)

enable_testing()

add_executable(sparse_layer_test tests/SparseLayerTest.cc)
target_link_libraries(
  sparse_layer_test 
  PRIVATE 
  bolt_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(sparse_layer_test)