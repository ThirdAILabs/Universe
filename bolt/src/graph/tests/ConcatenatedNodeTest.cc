
#include "MockNode.h"
#include <bolt/src/graph/nodes/Concatenate.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <algorithm>
#include <memory>

namespace thirdai::bolt::tests {

float gradientFromActiveNeuron(BoltVector& vector,
                               FoundActiveNeuron& active_neuron) {
  if (!active_neuron.pos) {
    return 0;
  }

  return vector.gradients[active_neuron.pos.value()];
}

uint32_t getExpectedNumNonzeros(const std::vector<uint32_t>& input_dense_dims,
                                const std::vector<BoltVector>& inputs,
                                bool sparse) {
  uint32_t expected_num_output_nonzeros = 0;
  for (uint32_t input_node_id = 0; input_node_id < input_dense_dims.size();
       input_node_id++) {
    const auto& input_vector = inputs.at(input_node_id);
    uint32_t input_dense_dim = input_dense_dims.at(input_node_id);

    expected_num_output_nonzeros += sparse ? input_vector.len : input_dense_dim;
  }
  return expected_num_output_nonzeros;
}

std::vector<NodePtr> getNodesToConcatenate(
    const std::vector<uint32_t>& input_dense_dims,
    const std::vector<BoltVector>& inputs) {
  std::vector<NodePtr> nodes_to_concatenate;
  for (uint32_t input_node_id = 0; input_node_id < input_dense_dims.size();
       input_node_id++) {
    uint32_t input_dense_dim = input_dense_dims.at(input_node_id);
    const auto& input_vector = inputs.at(input_node_id);
    nodes_to_concatenate.emplace_back(
        std::make_shared<MockNodeWithOutput>(input_vector, input_dense_dim));
  }
  return nodes_to_concatenate;
}

std::vector<uint32_t> getNeuronIndexOffsets(
    const std::vector<uint32_t>& input_dense_dims) {
  std::vector<uint32_t> neuron_index_offsets = {0};
  for (uint32_t input_dense_dim : input_dense_dims) {
    neuron_index_offsets.push_back(neuron_index_offsets.back() +
                                   input_dense_dim);
  }
  return neuron_index_offsets;
}

bool containsSparseVector(const std::vector<BoltVector>& inputs) {
  return std::any_of(inputs.begin(), inputs.end(),
                     [](const BoltVector& v) { return !v.isDense(); });
}

void testConcatForwardAndBackwardPass(
    const std::vector<uint32_t>& input_dense_dims,
    const std::vector<BoltVector>& inputs, bool sparse) {
  std::vector<NodePtr> nodes_to_concatenate =
      getNodesToConcatenate(input_dense_dims, inputs);
  uint32_t expected_num_output_nonzeros =
      getExpectedNumNonzeros(input_dense_dims, inputs, sparse);
  std::vector<uint32_t> neuron_index_offsets =
      getNeuronIndexOffsets(input_dense_dims);
  bool sparse_node_in_concatenation = containsSparseVector(inputs);

  // Use a shared pointer so shared_from_this() works
  std::shared_ptr<ConcatenateNode> concat_node =
      std::make_shared<ConcatenateNode>();
  concat_node->setConcatenatedNodes(nodes_to_concatenate);
  concat_node->initializeParameters();
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

  for (uint32_t input_node_id = 0; input_node_id < nodes_to_concatenate.size();
       input_node_id++) {
    uint32_t starting_output_index = neuron_index_offsets.at(input_node_id);
    uint32_t ending_output_index = neuron_index_offsets.at(input_node_id + 1);
    auto& current_input = nodes_to_concatenate.at(input_node_id)
                              ->getOutputVector(/* vec_index = */ 3);

    for (uint32_t output_index = starting_output_index;
         output_index < ending_output_index; output_index++) {
      auto output_neuron = output.findActiveNeuronNoTemplate(output_index);
      auto input_neuron = current_input.findActiveNeuronNoTemplate(
          output_index - starting_output_index);
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
      /* input_dense_dims = */ {2, 3, 1},
      /* inputs = */ {node_1_output, node_2_output, node_3_output},
      /* sparse = */ false);
  testConcatForwardAndBackwardPass(
      /* input_dense_dims = */ {2, 3, 1},
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
          /* input_dense_dims = */ {1000, 25, 2},
          /* inputs = */ {node_1_output, node_2_output, node_3_output},
          /* sparse = */ false),
      exceptions::NodeStateMachineError);
  testConcatForwardAndBackwardPass(
      /* input_dense_dims = */ {1000, 25, 2},
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
          /* input_dense_dims = */ {25, 3},
          /* inputs = */ {node_1_output, node_2_output},
          /* sparse = */ false),
      exceptions::NodeStateMachineError);
  testConcatForwardAndBackwardPass(
      /* input_dense_dims = */ {25, 3},
      /* inputs = */ {node_1_output, node_2_output},
      /* sparse = */ true);
}

}  // namespace thirdai::bolt::tests
