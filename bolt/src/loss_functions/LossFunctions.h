#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <_types/_uint32_t.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt {

class LossFunction {
 public:
  LossFunction() {}

  void lossGradients(BoltVector& output, const BoltVector& labels,
                     uint32_t batch_size) const {
    if (output.isDense()) {
      if (labels.isDense()) {
        computeLossGradientsImpl<true, true>(output, labels, batch_size);
      } else {
        computeLossGradientsImpl<true, false>(output, labels, batch_size);
      }
    } else {
      if (labels.isDense()) {
        computeLossGradientsImpl<false, true>(output, labels, batch_size);
      } else {
        computeLossGradientsImpl<false, false>(output, labels, batch_size);
      }
    }
  }

  float computeLossMetric(BoltVector& output, const BoltVector& labels) const {
    if (output.isDense()) {
      if (labels.isDense()) {
        return computeLossMetricImpl<true, true>(output, labels);
      }
      return computeLossMetricImpl<true, false>(output, labels);
    }
    if (labels.isDense()) {
      return computeLossMetricImpl<false, true>(output, labels);
    }
    return computeLossMetricImpl<false, false>(output, labels);
  }

  virtual ~LossFunction() = default;

 private:
  template <bool OUTPUT_DENSE, bool LABEL_DENSE>
  void computeLossGradientsImpl(BoltVector& output, const BoltVector& labels,
                                uint32_t batch_size) const {
    assert(!(OUTPUT_DENSE && output.active_neurons != nullptr));
    assert(!LABEL_DENSE || labels.active_neurons == nullptr);
    if (OUTPUT_DENSE && LABEL_DENSE) {
      assert(output.len == labels.len);
    }

    // Loss functions are only used in training.
    // If the label is sparse, the neurons of the network's final
    // layer that correspond to the label's nonzero elements are
    // automatically selected and activated during training.
    // Thus, we don't have to consider the case where there are
    // nonzeros in the label that correspond to inactive neurons in
    // the output layer.
    for (uint32_t i = 0; i < output.len; i++) {
      uint32_t active_neuron = OUTPUT_DENSE ? i : output.active_neurons[i];
      float label_val =
          labels.findActiveNeuron<LABEL_DENSE>(active_neuron).activation;
      output.gradients[i] =
          elementLossGradient(label_val, output.activations[i], batch_size);
    }
  }

  template <bool OUTPUT_DENSE, bool LABEL_DENSE>
  float computeLossMetricImpl(BoltVector& output,
                              const BoltVector& labels) const {
    float local_loss_metric_repeated = 0;
    float local_loss_metric_unique = 0;

    for (uint32_t i = 0; i < output.len; i++) {
      uint32_t active_neuron = OUTPUT_DENSE ? i : output.active_neurons[i];
      float label_val =
          labels.findActiveNeuron<LABEL_DENSE>(active_neuron).activation;
      if (label_val == 0) {
        local_loss_metric_unique +=
            elementLossMetric(label_val, output.activations[i]);
      } else {
        local_loss_metric_repeated +=
            elementLossMetric(label_val, output.activations[i]);
      }
    }

    for (uint32_t i = 0; i < labels.len; i++) {
      uint32_t active_neuron = LABEL_DENSE ? i : labels.active_neurons[i];
      float output_val =
          output.findActiveNeuron<OUTPUT_DENSE>(active_neuron).activation;
      if (output_val == 0) {
        local_loss_metric_unique +=
            elementLossMetric(labels.activations[i], output_val);
      } else {
        local_loss_metric_repeated +=
            elementLossMetric(labels.activations[i], output_val);
      }
    }
    float local_loss_metric =
        local_loss_metric_unique + local_loss_metric_repeated / 2;
    return local_loss_metric;
  }

  virtual float elementLossGradient(float label, float activation,
                                    uint32_t batch_size) const = 0;
  virtual float elementLossMetric(float label, float activation) const = 0;
};

class CategoricalCrossEntropyLoss final : public LossFunction {
 public:
  static std::shared_ptr<CategoricalCrossEntropyLoss>
  makeCategoricalCrossEntropyLoss() {
    return std::make_shared<CategoricalCrossEntropyLoss>();
  }

 private:
  float elementLossGradient(float label, float activation,
                            uint32_t batch_size) const override {
    return (label - activation) / batch_size;
  }
  float elementLossMetric(float label, float activation) const override {
    return -(label * log(activation));
  }
};

class MeanSquaredError final : public LossFunction {
 public:
  static std::shared_ptr<MeanSquaredError> makeMeanSquaredError() {
    return std::make_shared<MeanSquaredError>();
  }

 private:
  float elementLossGradient(float label, float activation,
                            uint32_t batch_size) const override {
    return 2 * (label - activation) / batch_size;
  }
  float elementLossMetric(float label, float activation) const override {
    return ((label - activation) * (label - activation));
  }
};

/**
 * This class computes gradients to optimize the model for weighted mean
 * absolute percentage error (WMAPE).
 * WMAPE is a regression error that measures the absolute deviation of
 * predictions from the true values, weighted in proportion to the true
 * values. WMAPE = 100% * sum(|actual - prediction|) / sum(|actual|)
 */
class WeightedMeanAbsolutePercentageErrorLoss final : public LossFunction {
 public:
  static std::shared_ptr<WeightedMeanAbsolutePercentageErrorLoss>
  makeWeightedMeanAbsolutePercentageErrorLoss() {
    return std::make_shared<WeightedMeanAbsolutePercentageErrorLoss>();
  }

 private:
  float elementLossGradient(float label, float activation,
                            uint32_t batch_size) const override {
    auto direction = activation > label ? -1.0 : 1.0;
    return direction / batch_size;
  }
  float elementLossMetric(float label, float activation) const override {
    throw std::invalid_argument(
        "Loss Metric is not supported for Weighted Mean Absolute Percentage "
        "Error Loss.");
    return ((label - activation) * (label - activation));
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
  if (lower_name == "meansquarederror" || lower_name == "mse") {
    return MeanSquaredError::makeMeanSquaredError();
  }
  if (lower_name == "weightedmeanabsolutepercentageerror" ||
      lower_name == "wmape") {
    return WeightedMeanAbsolutePercentageErrorLoss::
        makeWeightedMeanAbsolutePercentageErrorLoss();
  }
  throw std::invalid_argument(
      "'" + name +
      "' is not a valid loss function. Use 'CategoricalCrossEntropyLoss', "
      "'MeanSquaredError'/'MSE', or "
      "'WeightedMeanAbsolutePercentageError'/'WMAPE'");
}

}  // namespace thirdai::bolt
