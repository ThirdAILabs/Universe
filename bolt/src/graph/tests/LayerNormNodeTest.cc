
#include "MockNode.h"
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/LayerNorm.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace thirdai::bolt::tests {

static uint32_t n_classes = 100;
static float hidden_layer_sparsity = 0.25;

static BoltGraph buildSingleHiddenLayerModel(bool sparse_hidden_layer) {
  auto input = std::make_shared<Input>(/* expected_input_dim */ n_classes);
  auto hidden_layer = std::make_shared<FullyConnectedNode>(2000, 1.0, "ReLU");
  hidden_layer->addPredecessor(input);

  auto output = std::make_shared<FullyConnectedNode>(
      /* expected_dim */ n_classes, "Softmax");
  output->addPredecessor(hidden_layer);

  BoltGraph model({input}, output);
  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  if (sparse_hidden_layer) {
    LayerNameManager name_manager;
    hidden_layer->setSparsity(hidden_layer_sparsity);
  }

  return model;
}

// template <bool SPARSE>
// void testLayerNormNodeForwardPass() {
//   BoltGraph model = buildSingleHiddenLayerModel(SPARSE);

//   std::shared_ptr<LayerNormNode> layer_norm_node =
//       std::make_shared<LayerNormNode>();

//   layer_norm_node->setLayerNormNodeConfig(
//       /* center */ true,
//       /* scale */ true,
//       /* epsilon */ 0.001,
//       /* beta_regularizer */ 0.5,
//       /* gamma_regularizer */ 0.15,
//       /* beta_initializer */ 0,
//       /* gamma_initializer */ 1);

//   auto pred_node = model.getNodeByName("fc_1");
//   layer_norm_node->addPredecessor(pred_node);
//   LayerNameManager name_manager;

//   layer_norm_node->compile(name_manager);

//   layer_norm_node->prepareForBatchProcessing(/* batch_size = */ 32,
//                                              /*use_sparsity */ SPARSE);

//   auto pre_normalized_vector =
//       layer_norm_node->getOutputVector(/* vec_index */ 1);

//   layer_norm_node->forward(/* vec_index */ 1, /* labels */ nullptr);

//   BoltVector& output = layer_norm_node->getOutputVector(/* vec_index */ 1);

//   auto computed_moments = layer_norm_node->getMoments();
//   auto expected_moments = LayerNormNode::computeNormalizationMoments(
//       pre_normalized_vector, !SPARSE);

//   ASSERT_EQ(!output.isDense(), SPARSE);
//   ASSERT_EQ(pred_node->outputDim(), layer_norm_node->outputDim());

//   // Check that the mean for the activations is what's expected
//   ASSERT_FLOAT_EQ(computed_moments->first, expected_moments.first);

//   // Check that the variance for the activations is what's expected
//   ASSERT_FLOAT_EQ(computed_moments->second, expected_moments.second);
// }

// TEST(LayerNormNodeTest, DenseLayerNormalizationTest) {
//   testLayerNormNodeForwardPass<false>();
// }

// TEST(LayerNormNodeTest, SparseLayerNormalizationTest) {
//   testLayerNormNodeForwardPass<true>();
// }

}  // namespace thirdai::bolt::tests
