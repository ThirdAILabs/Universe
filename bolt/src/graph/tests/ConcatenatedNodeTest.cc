
#include "MockNode.h"
#include <bolt/src/graph/nodes/Concatenated.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>

using testing::Return;
using testing::ReturnRef;

namespace thirdai::bolt::tests {

std::shared_ptr<MockNode> getMockNodeWithOutput(BoltVector& output,
                                                uint32_t output_dim) {
  std::shared_ptr<MockNode> node = std::make_shared<MockNode>();
  EXPECT_CALL(*node, getOutputVector).WillRepeatedly(ReturnRef(output));
  EXPECT_CALL(*node, outputDim).WillRepeatedly(Return(output_dim));
  EXPECT_CALL(*node, numNonzerosInOutput).WillRepeatedly(Return(output.len));
  return node;
}

void testConcatForwardPass(std::vector<uint32_t> predecessor_output_dims,
                           std::vector<BoltVector> predecessor_outputs,
                           bool sparse) {
  std::vector<std::shared_ptr<Node>> nodes_to_concatenate;

  uint32_t expected_output_length = 0;
  bool sparse_node_in_concatenation = false;
  std::vector<uint32_t> label_offsets = {0};
  for (uint32_t concat_id = 0; concat_id < predecessor_output_dims.size();
       concat_id++) {
    uint32_t label_dim = predecessor_output_dims.at(concat_id);
    auto& output = predecessor_outputs.at(concat_id);
    nodes_to_concatenate.push_back(getMockNodeWithOutput(output, label_dim));
    sparse_node_in_concatenation |= !output.isDense();
    expected_output_length += sparse ? output.len : label_dim;
    label_offsets.push_back(label_offsets.back() + label_dim);
  }

  // Use make_shared to
  std::shared_ptr<ConcatenatedNode> concat_node =
      std::make_shared<ConcatenatedNode>();
  concat_node->setConcatenatedNodes(nodes_to_concatenate);
  concat_node->prepareForBatchProcessing(/* batch_size = */ 10,
                                         /* use_sparsity = */ sparse);
  concat_node->forward(/* vec_index = */ 3, /* labels = */ nullptr);
  auto& output = concat_node->getOutputVector(/* vec_index = */ 3);

  ASSERT_EQ(output.len, expected_output_length);
  bool outputDense = output.isDense();
  ASSERT_EQ(outputDense, !(sparse_node_in_concatenation && sparse));

  for (uint32_t concat_id = 0; concat_id < predecessor_output_dims.size();
       concat_id++) {
    uint32_t starting_label = label_offsets.at(concat_id);
    uint32_t ending_label = label_offsets.at(concat_id + 1);
    auto& current_output = predecessor_outputs.at(concat_id);
    for (uint32_t label = starting_label; label < ending_label; label++) {
      ASSERT_EQ(
          output.findActiveNeuronNoTemplate(label).activation,
          current_output.findActiveNeuronNoTemplate(label - starting_label)
              .activation);
    }
  }
}

TEST(ConcatenatedNodeTest, ForwardPassDenseConcatTest) {
  BoltVector node_1_output =
      BoltVector::makeDenseVector(/* values = */ {0.5, 0.75});
  BoltVector node_2_output =
      BoltVector::makeDenseVector(/* values = */ {0.25, 0, 0.25});
  BoltVector node_3_output = BoltVector::makeDenseVector(/* values = */ {0});
  testConcatForwardPass(
      /* predecessor_output_dims = */ {2, 3, 1},
      /* predecessor_outputs = */ {node_1_output, node_2_output, node_3_output},
      /* sparse = */ false);
  testConcatForwardPass(
      /* predecessor_output_dims = */ {2, 3, 1},
      /* predecessor_outputs = */ {node_1_output, node_2_output, node_3_output},
      /* sparse = */ true);
}

TEST(ConcatenatedNodeTest, ForwardPassSparseConcatTest) {
  BoltVector node_1_output = BoltVector::makeSparseVector(
      /* indices = */ {17, 3}, /* values = */ {0.5, 0.75});
  BoltVector node_2_output =
      BoltVector::makeSparseVector(/* indices = */ {1}, /* values = */ {0.25});
  testConcatForwardPass(
      /* predecessor_output_dims = */ {25, 2},
      /* predecessor_outputs = */ {node_1_output, node_2_output},
      /* sparse = */ false);
  testConcatForwardPass(
      /* predecessor_output_dims = */ {25, 2},
      /* predecessor_outputs = */ {node_1_output, node_2_output},
      /* sparse = */ true);
}

TEST(ConcatenatedNodeTest, ForwardPassSparseAndDenseConcatTest) {
  BoltVector node_1_output = BoltVector::makeSparseVector(
      /* indices = */ {17, 3}, /* values = */ {0.5, 0.75});
  BoltVector node_2_output =
      BoltVector::makeDenseVector(/* values = */ {0.25, 0, 0.25});
  testConcatForwardPass(
      /* predecessor_output_dims = */ {25, 3},
      /* predecessor_outputs = */ {node_1_output, node_2_output},
      /* sparse = */ false);
  testConcatForwardPass(
      /* predecessor_output_dims = */ {25, 3},
      /* predecessor_outputs = */ {node_1_output, node_2_output},
      /* sparse = */ true);
}

}  // namespace thirdai::bolt::tests
