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
  LossFunction() {
  }

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

  float computeLossMetric(BoltVector& output, const BoltVector& labels, 
                          uint32_t batch_size) const {
    assert(batch_size != 0);
    float local_loss_metric = 0;
    uint32_t labels_active_neuron_id = 0;
    uint32_t output_active_neuron_id = 0;
    // Assuming the neuron_id(indices) are in increasing order.
    while(labels_active_neuron_id<labels.len && output_active_neuron_id<output.len){
      if(labels.activations[labels_active_neuron_id]<output.activations[output_active_neuron_id]){
        local_loss_metric += elementLossMetric(labels.activations[labels_active_neuron_id], 0 , batch_size);
        labels_active_neuron_id++;
      }else if(labels.activations[labels_active_neuron_id]<output.activations[output_active_neuron_id]){
        local_loss_metric += elementLossMetric(0, output.activations[output_active_neuron_id] , batch_size);
        output_active_neuron_id++;
      }else{
        local_loss_metric += elementLossMetric(labels.activations[labels_active_neuron_id], output.activations[output_active_neuron_id] , batch_size);;
        labels_active_neuron_id++;
        output_active_neuron_id++;
      }
    }
    while(labels_active_neuron_id<labels.len){
        local_loss_metric += elementLossMetric(labels.activations[labels_active_neuron_id], 0 , batch_size);
        labels_active_neuron_id++;
    }
    while(output_active_neuron_id<output.len){
        local_loss_metric += elementLossMetric(0, output.activations[output_active_neuron_id] , batch_size);
        output_active_neuron_id++;
    }

    return local_loss_metric;
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
  

  virtual float elementLossGradient(float label, float activation,
                                    uint32_t batch_size) const = 0;
  virtual float elementLossMetric(float label, float activation,
                                    uint32_t batch_size) const = 0;

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
  float elementLossMetric(float label, float activation,
                            uint32_t batch_size) const override {
      assert(label > 0);
      return ((-1) * activation * log (label))/ batch_size;
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
  float elementLossMetric(float label, float activation,
                            uint32_t batch_size) const override {
    return ((label - activation) * (label - activation)) / batch_size;
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
  float elementLossMetric(float label, float activation,
                            uint32_t batch_size) const override {
    throw std::invalid_argument("Loss Metric is not supported for Weighted Mean Absolute Percentage Error Loss.");
    return ((label - activation) * (label - activation)) / batch_size;
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
