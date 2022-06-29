
#include "MockNode.h"
#include <bolt/src/graph/nodes/Concatenated.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>

using testing::ReturnRef;
using testing::Return;

namespace thirdai::bolt::tests {

TEST(ConcatenatedNodeTest, ForwardPassDenseConcatenationTest) {
  std::shared_ptr<MockNode> concatenated_node1 = std::make_shared<MockNode>();
  std::shared_ptr<MockNode> concatenated_node2 = std::make_shared<MockNode>();;
  BoltVector node_1_output =
      BoltVector::makeDenseVector(/* values = */ {0.5, 0.75});
  BoltVector node_2_output =
      BoltVector::makeDenseVector(/* values = */ {0.25, 0, 0.25});

  EXPECT_CALL(*concatenated_node1, getOutputVector)
      .WillRepeatedly(ReturnRef(node_1_output));
  EXPECT_CALL(*concatenated_node1, outputDim)
      .WillRepeatedly(Return(2));
  EXPECT_CALL(*concatenated_node1, numNonzerosInOutput)
      .WillRepeatedly(Return(2));

  EXPECT_CALL(*concatenated_node2, getOutputVector)
      .WillRepeatedly(ReturnRef(node_2_output));
  EXPECT_CALL(*concatenated_node2, outputDim)
      .WillRepeatedly(Return(3));
  EXPECT_CALL(*concatenated_node2, numNonzerosInOutput)
      .WillRepeatedly(Return(2));

  // This is a shared pointer instead of just an object so that shared_from_this
  // works in prepareForBatchProcessing.
  std::shared_ptr<ConcatenatedNode> concat_node = std::make_shared<ConcatenatedNode>();
  concat_node->setConcatenatedNodes({concatenated_node1, concatenated_node2});
  concat_node->prepareForBatchProcessing(/* batch_size = */ 1,
                                        /* use_sparsity = */ false);
  concat_node->forward(/* vec_index = */ 0, /* labels = */ nullptr);
  auto& output_node = concat_node->getOutputVector(/* vec_index = */ 0);

  ASSERT_EQ(output_node.len, node_1_output.len + node_2_output.len);
  ASSERT_TRUE(output_node.isDense());
  for (uint32_t i = 0; i < node_1_output.len; i++) {
    ASSERT_EQ(node_1_output.activations[i], output_node.activations[i]);
  }
  for (uint32_t i = 0; i < node_2_output.len; i++) {
    ASSERT_EQ(node_2_output.activations[i],
              output_node.activations[node_1_output.len + i]);
  }
}

}  // namespace thirdai::bolt::tests
