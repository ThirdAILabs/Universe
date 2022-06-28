#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <gtest/gtest.h>
#include <stdexcept>

namespace thirdai::bolt::tests {

TEST(GraphRejectsInvalidInputsTest, RejectInputLayerInOutput) {
  auto input_layer = std::make_shared<Input>(/* dim= */ 10);
  BoltGraph graph(/* inputs= */ {}, /* output= */ input_layer);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      graph.compile(std::make_shared<MeanSquaredError>()),
      std::invalid_argument);
}

TEST(GraphRejectsInvalidInputsTest,
     RejectSoftmaxWithoutCategoricalCrossEntropy) {
  auto layer = std::make_shared<FullyConnectedLayerNode>(
      /* dim= */ 10, /* activation= */ ActivationFunction::Softmax);
  BoltGraph graph(/* inputs= */ {}, /* output= */ layer);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      graph.compile(std::make_shared<MeanSquaredError>()),
      std::invalid_argument);
}

TEST(GraphRejectsInvalidInputsTest,
     RejectCategoricalCrossEntropyWithoutSoftmax) {
  auto layer = std::make_shared<FullyConnectedLayerNode>(
      /* dim= */ 10, /* activation= */ ActivationFunction::ReLU);
  BoltGraph graph(/* inputs= */ {}, /* output= */ layer);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      graph.compile(std::make_shared<CategoricalCrossEntropyLoss>()),
      std::invalid_argument);
}

TEST(GraphRejectsInvalidInputsTest, RejectBinaryCrossEntropyWithoutSigmoid) {
  auto layer = std::make_shared<FullyConnectedLayerNode>(
      /* dim= */ 10, /* activation= */ ActivationFunction::Softmax);
  BoltGraph graph(/* inputs= */ {}, /* output= */ layer);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      graph.compile(std::make_shared<BinaryCrossEntropyLoss>()),
      std::invalid_argument);
}

TEST(GraphRejectsInvalidInputsTest, AcceptsCategoricalCrossEntropyWithSoftmax) {
  auto layer = std::make_shared<FullyConnectedLayerNode>(
      /* dim= */ 10, /* activation= */ ActivationFunction::Softmax);
  BoltGraph graph(/* inputs= */ {}, /* output= */ layer);

  ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      graph.compile(std::make_shared<CategoricalCrossEntropyLoss>()));
}

TEST(GraphRejectsInvalidInputsTest, AcceptsBinaryCrossEntropyWithSigmoid) {
  auto layer = std::make_shared<FullyConnectedLayerNode>(
      /* dim= */ 10, /* activation= */ ActivationFunction::Sigmoid);
  BoltGraph graph(/* inputs= */ {}, /* output= */ layer);

  ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      graph.compile(std::make_shared<BinaryCrossEntropyLoss>()));
}

}  // namespace thirdai::bolt::tests