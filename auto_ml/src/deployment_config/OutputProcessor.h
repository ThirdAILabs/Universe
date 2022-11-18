#pragma once

#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/Categorical.h>
#include <optional>

namespace thirdai::automl::deployment {

class RegressionAsClassificationOutputProcessor {
 public:
  explicit RegressionAsClassificationOutputProcessor(
      dataset::RegressionCategoricalBlockPtr label_block)
      : _label_block(std::move(label_block)) {}

  bolt::InferenceOutputTracker processEvalOutput(
      bolt::InferenceOutputTracker& output) {
    std::vector<float> predicted_values(output.numSamples());

    for (uint32_t i = 0; i < output.numSamples(); i++) {
      uint32_t predicted_bin = getPredictedBin(output.activeNeuronsForSample(i),
                                               output.activationsForSample(i),
                                               output.numNonzerosInOutput());
      predicted_values[i] =
          _label_block->getDecimalValueForCategory(predicted_bin);
    }

    return bolt::InferenceOutputTracker(
        /* active_neurons= */ std::nullopt,
        /* activations= */ std::move(predicted_values),
        /* num_nonzeros_per_sample= */ output.numNonzerosInOutput());
  }

  BoltVector processVector(BoltVector& output) {
    uint32_t predicted_bin =
        getPredictedBin(output.active_neurons, output.activations, output.len);

    BoltVector value(/* l= */ 1, /* is_dense= */ true,
                     /* has_gradient= */ false);
    value.activations[0] =
        _label_block->getDecimalValueForCategory(predicted_bin);
    return value;
  }

 private:
  static uint32_t getPredictedBin(const uint32_t* active_neurons,
                                  const float* activations, uint32_t len) {
    uint32_t predicted_bin_index = 0;
    float max_activation = activations[0];

    for (uint32_t i = 1; i < len; i++) {
      if (activations[i] > max_activation) {
        predicted_bin_index = i;
        max_activation = activations[i];
      }
    }

    if (active_neurons != nullptr) {
      return active_neurons[predicted_bin_index];
    }
    return predicted_bin_index;
  }

  dataset::RegressionCategoricalBlockPtr _label_block;
};

}  // namespace thirdai::automl::deployment