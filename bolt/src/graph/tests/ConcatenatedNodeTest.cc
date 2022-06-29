
#include "MockNode.h"
#include <bolt/src/graph/nodes/Concatenated.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>

using testing::ReturnPointee;

namespace thirdai::bolt::tests {

TEST(ConcatenatedNodeTest, ForwardPassDenseConcatenationTest) {
  std::shared_ptr<MockNode> concatenated_node1;
  std::shared_ptr<MockNode> concatenated_node2;
  BoltVector node_1_output =
      BoltVector::makeDenseVector(/* values = */ {0.5, 0.5, 0.5});
  BoltVector node_2_output =
      BoltVector::makeDenseVector(/* values = */ {0.25, 0.25, 0.25});
  ON_CALL(*concatenated_node1, getOutputVector)
      .WillByDefault(ReturnPointee(&node_1_output));
  ON_CALL(*concatenated_node2, getOutputVector)
      .WillByDefault(ReturnPointee(&node_2_output));

  ConcatenatedNode concat_node;
  concat_node.setConcatenationInputs({concatenated_node1, concatenated_node2});

  concat_node.prepareForBatchProcessing(/* batch_size = */ 1,
                                        /* use_sparsity = */ false);
  concat_node.forward(/* vec_index = */ 0, /* labels = */ nullptr);
  auto& output_node = concat_node.getOutputVector(/* vec_index = */ 0);

  ASSERT_EQ(output_node.len, node_1_output.len + node_2_output.len);
  ASSERT_TRUE(output_node.isDense());
  for (uint32_t i = 0; i < node_1_output.len; i++) {
    ASSERT_EQ(node_1_output.activations[i], output_node.activations[i]);
  }
  for (uint32_t i = 0; i < node_2_output.len; i++) {
    ASSERT_EQ(node_1_output.activations[i],
              output_node.activations[node_1_output.len + i]);
  }
}

}  // namespace thirdai::bolt::tests
