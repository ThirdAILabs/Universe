
#include "MockNode.h"
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/LayerNorm.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/networks/tests/BoltNetworkTestUtils.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>

namespace thirdai::bolt::tests {

static uint32_t n_classes = 100;
static uint32_t batch_size = 32;

using NodeGraphTuple = std::tuple<std::shared_ptr<LayerNormNode>, BoltGraph>;

static NodeGraphTuple buildSingleNormNodeModel() {
  auto input = std::make_shared<Input>(/* expected_input_dim */ n_classes);
  auto hidden_layer = std::make_shared<FullyConnectedNode>(2000, 1.0, "ReLU");
  hidden_layer->addPredecessor(input);

  NormalizationLayerConfig layer_norm_config =
      NormalizationLayerConfig::makeConfig();
  auto normalization_layer = std::make_shared<LayerNormNode>(layer_norm_config);
  normalization_layer->addPredecessor(hidden_layer);

  auto output = std::make_shared<FullyConnectedNode>(
      /* expected_dim */ n_classes, "Softmax");
  output->addPredecessor(normalization_layer);

  BoltGraph model({input}, output);
  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  return std::make_tuple(normalization_layer, model);
}

NormalizationLayerConfig& getLayerNormConfig() {
  return NormalizationLayerConfig::makeConfig()
      .setCenteringFactor(/*centering_factor= */ 0.0)
      .setScalingFactor(/* scaling_factor= */ 1.0);
}

NodePtr getInputVector(uint32_t length) {
  std::vector<float> values(length);
  std::vector<uint32_t> active_neurons(length);

  std::random_device rd;
  std::default_random_engine generator(rd());
  std::uniform_real_distribution<float> distribution(1.0, 10.0);

  for (uint32_t index = 0; index < length; index++) {
    active_neurons.push_back(index);
    float random_activation = distribution(generator);
    values.push_back(random_activation);
  }
  auto output = BoltVector(&active_neurons[0], &values[0], nullptr, length);

  return std::make_shared<MockNodeWithOutput>(&output, length);
}

void testLayerNormNodeForwardAndBackwardPass2(bool sparse) {

  NodePtr input_node = getInputVector(/* length= */ 10);
  std::shared_ptr<LayerNormNode> layer_norm_node =
      std::make_shared<LayerNormNode>(getLayerNormConfig());
  
  layer_norm_node->addPredecessor(input_node);
  LayerNameManager name_manager;

  ASSERT_EQ(input_node->outputDim(), layer_norm_node->outputDim());

  layer_norm_node->compile(name_manager);

  layer_norm_node->prepareForBatchProcessing(/* batch_size= */ 5,
                                             /* use_sparsity= */ sparse);
  layer_norm_node->forward(/* vec_index= */ 0, /* labels= */ nullptr);

  auto& output_vector = layer_norm_node->getOutputVector(/* vec_index= */ 0);

  layer_norm_node->backpropagate(/* vec_index= */ 0);

  for (uint32_t neuron_index = 0; neuron_index < output_vector.len;
       neuron_index++) {
    ASSERT_FLOAT_EQ(output_vector.activations[neuron_index], 1.0);
  }
}

void testLayerNormNodeForwardAndBackwardPass() {
  auto [layer_norm_node, model] = buildSingleNormNodeModel();

  auto [data, labels] =
      genDataset(/* n_classes= */ n_classes, /* noisy_dataset= */ false);
  TrainConfig train_config =
      TrainConfig::makeConfig(/* learning_rate= */ 0.001, /* epochs= */ 5)
          .withMetrics({"mean_squared_error"})
          .withBatchSize(batch_size)
          .silence();

  model.train(/* train_data= */ {data}, /* train_tokens= */ {}, labels,
              train_config);

  auto pred_node = model.getNodeByName("layer_norm_1")->getPredecessors()[0];
  ASSERT_EQ(pred_node->outputDim(),
            model.getNodeByName("layer_norm_1")->outputDim());

  // BoltVector& output_vector =
  //     layer_norm_node->getOutputVector(/* vec_index= */ 2);

  // BoltVector& input_vector = model.getNodeByName("fc_1")->getOutputVector(/*
  // vec_index= */ 1);

  // for (uint32_t i = 0; i < input_vector.len; i++) {
  //   std::cout << " input_grad = " << input_vector.gradients[i] << std::endl;
  // }

  // for (uint32_t i = 0; i < output_vector.len; i++) {
  //   std::cout << " output_grad = " << output_vector.gradients[i] <<
  //   std::endl;
  // }

  // for (uint32_t neuron_index = 0; neuron_index < output_vector.len;
  //      neuron_index++) {
  //   ASSERT_NE(output_vector.gradients[neuron_index], 0.0);
  // }
}

TEST(LayerNormNodeTest, LayerNormalizationTest) {
  testLayerNormNodeForwardAndBackwardPass();
}

}  // namespace thirdai::bolt::tests
