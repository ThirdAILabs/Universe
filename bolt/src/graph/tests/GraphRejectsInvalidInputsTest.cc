#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <gtest/gtest.h>
#include <exceptions/src/Exceptions.h>
#include <memory>

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
  auto layer = std::make_shared<FullyConnectedNode>(
      /* dim= */ 10, /* activation= */ "softmax");
  BoltGraph graph(/* inputs= */ {}, /* output= */ layer);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      graph.compile(std::make_shared<MeanSquaredError>()),
      exceptions::GraphCompilationFailure);
}

TEST(GraphRejectsInvalidInputsTest,
     RejectCategoricalCrossEntropyWithoutSoftmax) {
  auto layer = std::make_shared<FullyConnectedNode>(
      /* dim= */ 10, /* activation= */ "relu");
  BoltGraph graph(/* inputs= */ {}, /* output= */ layer);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      graph.compile(std::make_shared<CategoricalCrossEntropyLoss>()),
      exceptions::GraphCompilationFailure);
}

TEST(GraphRejectsInvalidInputsTest, RejectBinaryCrossEntropyWithoutSigmoid) {
  auto layer = std::make_shared<FullyConnectedNode>(
      /* dim= */ 10, /* activation= */ "softmax");
  BoltGraph graph(/* inputs= */ {}, /* output= */ layer);

  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      graph.compile(std::make_shared<BinaryCrossEntropyLoss>()),
      exceptions::GraphCompilationFailure);
}

TEST(GraphRejectsInvalidInputsTest, AcceptsCategoricalCrossEntropyWithSoftmax) {
  auto input = std::make_shared<Input>(/* dim= */ 10);
  auto layer = std::make_shared<FullyConnectedNode>(
      /* dim= */ 10, /* activation= */ "softmax");
  layer->addPredecessor(input);

  BoltGraph graph(/* inputs= */ {input}, /* output= */ layer);

  ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      graph.compile(std::make_shared<CategoricalCrossEntropyLoss>()));
}

TEST(GraphRejectsInvalidInputsTest, AcceptsBinaryCrossEntropyWithSigmoid) {
  auto input = std::make_shared<Input>(/* dim= */ 10);
  auto layer = std::make_shared<FullyConnectedNode>(
      /* dim= */ 10, /* activation= */ "sigmoid");
  layer->addPredecessor(input);

  BoltGraph graph(/* inputs= */ {input}, /* output= */ layer);

  ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      graph.compile(std::make_shared<BinaryCrossEntropyLoss>()));
}

TEST(GraphRejectsInvalidInputsTest, RejectConcatenatingInputLayer) {
  auto input_layer = std::make_shared<Input>(/* dim= */ 10);
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      std::make_shared<ConcatenateNode>()->setConcatenatedNodes(
          {input_layer, input_layer}),
      exceptions::GraphCompilationFailure);
}

TEST(GraphRejectsInvalidInputsTest, RejectConcatenateAsOutputLayer) {
  auto input = std::make_shared<Input>(/* dim= */ 10);
  auto layer = std::make_shared<FullyConnectedNode>(
                   /* dim= */ 10, /* activation= */ "relu")
                   ->addPredecessor(input);
  auto concat =
      std::make_shared<ConcatenateNode>()->setConcatenatedNodes({layer, layer});
  BoltGraph graph(/* inputs= */ {input}, /* output= */ concat);
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      graph.compile(std::make_shared<MeanSquaredError>()),
      exceptions::GraphCompilationFailure);
}

TEST(GraphRejectsInvalidInputsTest, AcceptsCorrectConcatenation) {
  auto input = std::make_shared<Input>(/* dim= */ 10);
  auto layer_1 = std::make_shared<FullyConnectedNode>(
                     /* dim= */ 10, /* activation= */ "relu")
                     ->addPredecessor(input);
  auto layer_2 = std::make_shared<FullyConnectedNode>(
                     /* dim= */ 10, /* activation= */ "relu")
                     ->addPredecessor(input);
  auto concat_1 = std::make_shared<ConcatenateNode>()->setConcatenatedNodes(
      {layer_1, layer_2, layer_2});
  auto layer_3 = std::make_shared<FullyConnectedNode>(
                     /* dim= */ 10, /* activation= */ "relu")
                     ->addPredecessor(concat_1);
  auto concat_2 = std::make_shared<ConcatenateNode>()->setConcatenatedNodes(
      {layer_1, layer_3, concat_1});
  auto output = std::make_shared<FullyConnectedNode>(
                    /* dim= */ 10, /* activation= */ "relu")
                    ->addPredecessor(concat_2);
  BoltGraph graph(/* inputs= */ {input}, /* output= */ output);
  ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      graph.compile(std::make_shared<MeanSquaredError>()));
}

TEST(GraphRejectsInvalidInputsTest, RejectsUnkownInput) {
  auto input1 = std::make_shared<Input>(/* dim= */ 10);
  auto input2 = std::make_shared<Input>(/* dim= */ 20);

  auto layer = std::make_shared<FullyConnectedNode>(
                   /* dim= */ 10, /* activation= */ "relu")
                   ->addPredecessor(input1);

  BoltGraph graph(/* inputs= */ {input2}, /* output= */ layer);
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      graph.compile(std::make_shared<MeanSquaredError>()),
      exceptions::GraphCompilationFailure);
}

TEST(GraphRejectsInvalidInputsTest, RejectsUnusedInput) {
  auto input1 = std::make_shared<Input>(/* dim= */ 10);
  auto input2 = std::make_shared<Input>(/* dim= */ 20);

  auto layer = std::make_shared<FullyConnectedNode>(
                   /* dim= */ 10, /* activation= */ "relu")
                   ->addPredecessor(input1);

  BoltGraph graph(/* inputs= */ {input1, input2}, /* output= */ layer);
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
      graph.compile(std::make_shared<MeanSquaredError>()),
      exceptions::GraphCompilationFailure);
}

}  // namespace thirdai::bolt::tests