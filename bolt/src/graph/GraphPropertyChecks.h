#pragma once

#include "Node.h"
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/graph/nodes/LayerNorm.h>
#include <bolt/src/loss_functions/LossFunctions.h>

namespace thirdai::bolt {

class GraphPropertyChecks {
 public:
  static void verifyOutputIsNotInputLayer(const NodePtr& output) {
    if (output->isInputNode()) {
      throw exceptions::GraphCompilationFailure(
          "Output node cannot be an input node.");
    }
  }

  static void verifyOutputIsFullyConnectedLayer(const NodePtr& output) {
    if (dynamic_cast<ConcatenateNode*>(output.get())) {
      throw exceptions::GraphCompilationFailure(
          "Output node cannot be a Concatenate node.");
    }

    // For clarity, the output can be normalized. There is nothing in the
    // implementation that prohibits a normalization layer from being the
    // output layer. But for best practices, oftentimes it doesn't make
    // sense to normalize the outputs (for instance if the outputs are the
    // results of applying the softmax activation, which yields values in
    // [0, 1] interval).
    if (dynamic_cast<LayerNormNode*>(output.get())) {
      throw exceptions::GraphCompilationFailure(
          "Output node cannot be a normalization node");
    }
  }

  static void verifySoftmaxIsUsedWithCategoricalCrossEntropy(
      const NodePtr& output, const std::shared_ptr<LossFunction>& loss) {
    FullyConnectedNode* fc_output =
        dynamic_cast<FullyConnectedNode*>(output.get());

    if (fc_output != nullptr) {
      bool is_categorical_cross_entropy =
          dynamic_cast<CategoricalCrossEntropyLoss*>(loss.get()) != nullptr;
      bool is_softmax =
          fc_output->getActivationFunction() == ActivationFunction::Softmax;
      if ((is_categorical_cross_entropy && !is_softmax) ||
          (is_softmax && !is_categorical_cross_entropy)) {
        throw exceptions::GraphCompilationFailure(
            "Softmax activation must be used with categorical cross entropy "
            "loss.");
      }
    }
  }

  static void verifySigmoidIsUsedWithBinaryCrossEntropy(
      const NodePtr& output, const std::shared_ptr<LossFunction>& loss) {
    FullyConnectedNode* fc_output =
        dynamic_cast<FullyConnectedNode*>(output.get());

    if (fc_output != nullptr) {
      bool is_binary_cross_entropy =
          dynamic_cast<BinaryCrossEntropyLoss*>(loss.get()) != nullptr;
      bool is_sigmoid =
          fc_output->getActivationFunction() == ActivationFunction::Sigmoid;
      if ((is_binary_cross_entropy && !is_sigmoid) ||
          (is_sigmoid && !is_binary_cross_entropy)) {
        throw exceptions::GraphCompilationFailure(
            "Sigmoid activation must be used with binary cross entropy loss.");
      }
    }
  }
};

}  // namespace thirdai::bolt