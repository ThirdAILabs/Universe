#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt {

class LossFunction {
 public:
  LossFunction() {}

  virtual void loss(BoltVector& output, const BoltVector& labels,
                    uint32_t batch_size) const = 0;

  /**
   * Lambda type is templated because this helps the compiler inline
   * the lambda call.
   * https://stackoverflow.com/questions/13722426/why-can-lambdas-be-better-optimized-by-the-compiler-than-plain-functions
   */
  template <typename F>
  void computeLoss(BoltVector& output, const BoltVector& labels,
                   F element_loss) const {
    if (output.isDense()) {
      if (labels.isDense()) {
        computeLossImpl<true, true>(output, labels, element_loss);
      } else {
        computeLossImpl<true, false>(output, labels, element_loss);
      }
    } else {
      if (labels.isDense()) {
        computeLossImpl<false, true>(output, labels, element_loss);
      } else {
        computeLossImpl<false, false>(output, labels, element_loss);
      }
    }
  }

  virtual ~LossFunction() = default;

 private:
  /**
   * Lambda type is templated because this helps the compiler inline
   * the lambda call.
   * https://stackoverflow.com/questions/13722426/why-can-lambdas-be-better-optimized-by-the-compiler-than-plain-functions
   */
  template <bool OUTPUT_DENSE, bool LABEL_DENSE, typename F>
  void computeLossImpl(BoltVector& output, const BoltVector& labels,
                       F element_loss) const {
    assert(!OUTPUT_DENSE || output.active_neurons == nullptr);
    assert(!LABEL_DENSE || labels.active_neurons == nullptr);
    if (OUTPUT_DENSE && LABEL_DENSE) {
      assert(output.len == labels.len);
    }

    // Loss functions are only used in training.
    // Since active neurons in labels are automatically included in
    // the final layer's active neurons during training, we don't
    // have to consider the case where there are active neurons in
    // labels that are not a subset of output's active neurons.
    for (uint32_t i = 0; i < output.len; i++) {
      uint32_t active_neuron = OUTPUT_DENSE ? i : output.active_neurons[i];
      float label_val;
      if (LABEL_DENSE) {
        label_val = labels.activations[active_neuron];
      } else {
        const uint32_t* label_start = labels.active_neurons;
        const uint32_t* label_end = labels.active_neurons + labels.len;
        const uint32_t* itr = std::find(label_start, label_end, active_neuron);
        if (itr == label_end) {
          label_val = 0.0;
        } else {
          label_val = labels.activations[std::distance(label_start, itr)];
        }
      }

      output.gradients[i] = element_loss(label_val, output.activations[i]);
    }
  }
};

class CategoricalCrossEntropyLoss final : public LossFunction {
 public:
  static std::shared_ptr<CategoricalCrossEntropyLoss>
  makeCategoricalCrossEntropyLoss() {
    return std::make_shared<CategoricalCrossEntropyLoss>();
  }

  void loss(BoltVector& output, const BoltVector& labels,
            uint32_t batch_size) const override {
    computeLoss(output, labels, [&](float label, float activation) {
      return (label - activation) / batch_size;
    });
  }
};

class MeanSquaredError final : public LossFunction {
 public:
  static std::shared_ptr<MeanSquaredError> makeMeanSquaredError() {
    return std::make_shared<MeanSquaredError>();
  }

  void loss(BoltVector& output, const BoltVector& labels,
            uint32_t batch_size) const override {
    computeLoss(output, labels, [&](float label, float activation) {
      return 2 * (label - activation) / batch_size;
    });
  }
};

class WeightedMeanAbsolutePercentageErrorLoss final : public LossFunction {
 public:
  static std::shared_ptr<WeightedMeanAbsolutePercentageErrorLoss>
  makeWeightedMeanAbsolutePercentageErrorLoss() {
    return std::make_shared<WeightedMeanAbsolutePercentageErrorLoss>();
  }

  void loss(BoltVector& output, const BoltVector& labels,
            uint32_t batch_size) const override {
    float sum_of_squared_truth_elems = 0.0;
    for (uint32_t i = 0; i < labels.len; i++) {
      sum_of_squared_truth_elems +=
          labels.activations[i] * labels.activations[i];
    }
    float almost_zero = 0.0000001;
    float abs_truth =
        std::max(std::sqrt(sum_of_squared_truth_elems), almost_zero);
    computeLoss(output, labels, [&](float label, float activation) {
      auto factor = activation == label ? 0.0 : activation > label ? -1.0 : 1.0;
      return factor / (abs_truth * batch_size);
    });
  }
};

static std::shared_ptr<LossFunction> getLossFunction(const std::string& name) {
  std::string lower_name;
  for (char c : name) {
    lower_name.push_back(std::tolower(c));
  }
  if (lower_name == "categoricalcrossentropyloss") {
    return CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss();
  }
  if (lower_name == "meansquarederror") {
    return MeanSquaredError::makeMeanSquaredError();
  }
  if (lower_name == "weightedmeanabsolutepercentageerror") {
    return WeightedMeanAbsolutePercentageErrorLoss::
        makeWeightedMeanAbsolutePercentageErrorLoss();
  }
  throw std::invalid_argument(
      "'" + name +
      "' is not a valid loss function. Use CategoricalCrossEntropyLoss, "
      "MeanSquaredError, or WeightedMeanAbsolutePercentageError");
}

}  // namespace thirdai::bolt
