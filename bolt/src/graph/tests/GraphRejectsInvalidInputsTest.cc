#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <gtest/gtest.h>
#include <exceptions/src/Exceptions.h>

namespace thirdai::bolt::tests {

TEST(GraphRejectsInvalidInputsTest, RejectInputLayerInOutput) {
  auto input_layer = std::make_shared<Input>(/* dim= */ 10);
  BoltGraph graph(/* inputs= */ {}, /* output= */ input_layer);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      graph.compile(std::make_shared<MeanSquaredError>()),
      exceptions::GraphCompilationFailure);
}

TEST(GraphRejectsInvalidInputsTest,
     RejectSoftmaxWithoutCategoricalCrossEntropy) {
  auto layer = std::make_shared<FullyConnectedLayerNode>(
      /* dim= */ 10, /* activation= */ ActivationFunction::Softmax);
  BoltGraph graph(/* inputs= */ {}, /* output= */ layer);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      graph.compile(std::make_shared<MeanSquaredError>()),
      exceptions::GraphCompilationFailure);
}

TEST(GraphRejectsInvalidInputsTest,
     RejectCategoricalCrossEntropyWithoutSoftmax) {
  auto layer = std::make_shared<FullyConnectedLayerNode>(
      /* dim= */ 10, /* activation= */ ActivationFunction::ReLU);
  BoltGraph graph(/* inputs= */ {}, /* output= */ layer);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      graph.compile(std::make_shared<CategoricalCrossEntropyLoss>()),
      exceptions::GraphCompilationFailure);
}

TEST(GraphRejectsInvalidInputsTest, RejectBinaryCrossEntropyWithoutSigmoid) {
  auto layer = std::make_shared<FullyConnectedLayerNode>(
      /* dim= */ 10, /* activation= */ ActivationFunction::Softmax);
  BoltGraph graph(/* inputs= */ {}, /* output= */ layer);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      graph.compile(std::make_shared<BinaryCrossEntropyLoss>()),
      exceptions::GraphCompilationFailure);
}

TEST(GraphRejectsInvalidInputsTest, AcceptsCategoricalCrossEntropyWithSoftmax) {
  auto input = std::make_shared<Input>(/* dim= */ 10);
  auto layer = std::make_shared<FullyConnectedLayerNode>(
      /* dim= */ 10, /* activation= */ ActivationFunction::Softmax);
  layer->addPredecessor(input);

  BoltGraph graph(/* inputs= */ {input}, /* output= */ layer);

  ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      graph.compile(std::make_shared<CategoricalCrossEntropyLoss>()));
}

TEST(GraphRejectsInvalidInputsTest, AcceptsBinaryCrossEntropyWithSigmoid) {
  auto input = std::make_shared<Input>(/* dim= */ 10);
  auto layer = std::make_shared<FullyConnectedLayerNode>(
      /* dim= */ 10, /* activation= */ ActivationFunction::Sigmoid);
  layer->addPredecessor(input);

  BoltGraph graph(/* inputs= */ {input}, /* output= */ layer);

  ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      graph.compile(std::make_shared<BinaryCrossEntropyLoss>()));
}

}  // namespace thirdai::bolt::tests