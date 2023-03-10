#include "MockNode.h"
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/Sparsification.h>
#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>

namespace thirdai::bolt::tests {

TEST(SparsificationTest, CorrectNonzeros) {
  BoltVector dense_vec = BoltVector::makeDenseVectorWithGradients(
      {-1.0, 4.0, 2.0, 2.5, -3.0, 1.5});

  auto mock_input =
      std::make_shared<MockNodeWithOutput>(dense_vec, dense_vec.len);

  auto sparsification = SparsificationNode::make(/* sparsity= */ 0.5);
  sparsification->addPredecessor(mock_input);

  LayerNameManager name_manager;
  sparsification->compile(name_manager);

  sparsification->prepareForBatchProcessing(/* batch_size= */ 1,
                                            /* use_sparsity= */ true);

  sparsification->forward(/* vec_index= */ 0, /* labels= */ nullptr);

  std::vector<std::pair<uint32_t, float>> expected_nonzeros = {
      {3, 2.5}, {4, -3.0}, {1, 4.0}};

  BoltVector& output_vec = sparsification->getOutputVector(/* vec_index= */ 0);
  ASSERT_EQ(output_vec.len, 3);

  for (uint32_t i = 0; i < 3; i++) {
    ASSERT_EQ(output_vec.active_neurons[i], expected_nonzeros[i].first);
    ASSERT_EQ(output_vec.activations[i], expected_nonzeros[i].second);
  }

  std::vector<float> output_gradients = {6.0, 7.0, 8.0};
  std::copy(output_gradients.begin(), output_gradients.end(),
            output_vec.gradients);

  sparsification->backpropagate(/* vec_index= */ 0);

  std::vector<float> expected_input_gradients = {0, 8.0, 0, 6.0, 7.0, 0};

  for (uint32_t i = 0; i < expected_input_gradients.size(); i++) {
    ASSERT_EQ(mock_input->getOutputVectorImpl(0).gradients[i],
              expected_input_gradients[i]);
  }
}

}  // namespace thirdai::bolt::tests