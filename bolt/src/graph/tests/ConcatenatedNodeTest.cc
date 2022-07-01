
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

float gradientFromActiveNeuron(BoltVector& vector,
                               FoundActiveNeuron& active_neuron) {
  if (!active_neuron.pos) {
    return 0;
  }

  return vector.gradients[active_neuron.pos.value()];
}

void testConcatForwardAndBackwardPass(std::vector<uint32_t> input_label_dims,
                                      std::vector<BoltVector> inputs,
                                      bool sparse) {
  std::vector<NodePtr> nodes_to_concatenate;

  uint32_t expected_num_output_nonzeros = 0;
  bool sparse_node_in_concatenation = false;
  std::vector<uint32_t> label_offsets = {0};
  for (uint32_t input_node_id = 0; input_node_id < input_label_dims.size();
       input_node_id++) {
    uint32_t label_dim = input_label_dims.at(input_node_id);
    auto& input_vector = inputs.at(input_node_id);
    nodes_to_concatenate.push_back(
        getMockNodeWithOutput(input_vector, label_dim));
    sparse_node_in_concatenation |= !input_vector.isDense();
    expected_num_output_nonzeros += sparse ? input_vector.len : label_dim;
    label_offsets.push_back(label_offsets.back() + label_dim);
  }

  // Use a shared pointer so shared_from_this() works
  std::shared_ptr<ConcatenatedNode> concat_node =
      std::make_shared<ConcatenatedNode>();
  concat_node->setConcatenatedNodes(nodes_to_concatenate);
  concat_node->prepareForBatchProcessing(/* batch_size = */ 10,
                                         /* use_sparsity = */ sparse);
  concat_node->forward(/* vec_index = */ 3, /* labels = */ nullptr);
  auto& output = concat_node->getOutputVector(/* vec_index = */ 3);

  ASSERT_EQ(output.len, expected_num_output_nonzeros);
  bool outputDense = output.isDense();
  bool outputShouldBeSparse = sparse_node_in_concatenation && sparse;
  ASSERT_EQ(outputDense, !outputShouldBeSparse);

  for (uint32_t i = 0; i < output.len; i++) {
    // The value of the gradients don't matter at all, we just want to make sure
    // they are being backpropogated correctly
    output.gradients[i] = rand();
  }
  concat_node->backpropagate(/* vec_index = */ 3);

  for (uint32_t input_node_id = 0; input_node_id < input_label_dims.size();
       input_node_id++) {
    uint32_t starting_label = label_offsets.at(input_node_id);
    uint32_t ending_label = label_offsets.at(input_node_id + 1);
    auto& current_input = inputs.at(input_node_id);

    for (uint32_t label = starting_label; label < ending_label; label++) {
      auto output_neuron = output.findActiveNeuronNoTemplate(label);
      auto input_neuron =
          current_input.findActiveNeuronNoTemplate(label - starting_label);
      ASSERT_EQ(input_neuron.activation, output_neuron.activation);
      float input_gradient =
          gradientFromActiveNeuron(current_input, input_neuron);
      float output_gradient = gradientFromActiveNeuron(output, output_neuron);
      ASSERT_EQ(input_gradient, output_gradient);
    }
  }
}

TEST(ConcatenatedNodeTest, DenseConcatTest) {
  BoltVector node_1_output =
      BoltVector::makeDenseVectorWithGradients(/* values = */ {0.5, 0.75});
  BoltVector node_2_output =
      BoltVector::makeDenseVectorWithGradients(/* values = */ {0.25, 0, 0.25});
  BoltVector node_3_output =
      BoltVector::makeDenseVectorWithGradients(/* values = */ {0});
  testConcatForwardAndBackwardPass(
      /* input_label_dims = */ {2, 3, 1},
      /* inputs = */ {node_1_output, node_2_output, node_3_output},
      /* sparse = */ false);
  testConcatForwardAndBackwardPass(
      /* input_label_dims = */ {2, 3, 1},
      /* inputs = */ {node_1_output, node_2_output, node_3_output},
      /* sparse = */ true);
}

TEST(ConcatenatedNodeTest, SparseConcatTest) {
  BoltVector node_1_output = BoltVector::makeSparseVectorWithGradients(
      /* indices = */ {100}, /* values = */ {0.25});
  BoltVector node_2_output = BoltVector::makeSparseVectorWithGradients(
      /* indices = */ {17, 3}, /* values = */ {0.5, 0.75});
  BoltVector node_3_output = BoltVector::makeSparseVectorWithGradients(
      /* indices = */ {1}, /* values = */ {0.25});
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      testConcatForwardAndBackwardPass(
          /* input_label_dims = */ {1000, 25, 2},
          /* inputs = */ {node_1_output, node_2_output, node_3_output},
          /* sparse = */ false),
      exceptions::NodeStateMachineError);
  testConcatForwardAndBackwardPass(
      /* input_label_dims = */ {1000, 25, 2},
      /* inputs = */ {node_1_output, node_2_output, node_3_output},
      /* sparse = */ true);
}

TEST(ConcatenatedNodeTest, SparseAndDenseConcatTest) {
  BoltVector node_1_output = BoltVector::makeSparseVectorWithGradients(
      /* indices = */ {17, 3}, /* values = */ {0.5, 0.75});
  BoltVector node_2_output =
      BoltVector::makeDenseVectorWithGradients(/* values = */ {0.25, 0, 0.25});
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      testConcatForwardAndBackwardPass(
          /* input_label_dims = */ {25, 3},
          /* inputs = */ {node_1_output, node_2_output},
          /* sparse = */ false),
      exceptions::NodeStateMachineError);
  testConcatForwardAndBackwardPass(
      /* input_label_dims = */ {25, 3},
      /* inputs = */ {node_1_output, node_2_output},
      /* sparse = */ true);
}

}  // namespace thirdai::bolt::tests
