#pragma once

#include <cereal/types/polymorphic.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <algorithm>
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
    /*
      Loss functions are only used in training.
      If the label is sparse, the neurons of the network's final
      layer that correspond to the label's nonzero elements are
      automatically selected and activated during training.
      Thus, we don't have to consider the case where there are
      nonzeros in the label that correspond to inactive neurons in
      the output layer.
    */
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

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

class CategoricalCrossEntropyLoss final : public LossFunction {
 public:
  CategoricalCrossEntropyLoss() {}

  static std::shared_ptr<CategoricalCrossEntropyLoss>
  makeCategoricalCrossEntropyLoss() {
    return std::make_shared<CategoricalCrossEntropyLoss>();
  }

 private:
  float elementLossGradient(float label, float activation,
                            uint32_t batch_size) const final {
    return (label - activation) / batch_size;
  }

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<LossFunction>(this));
  }
};

class BinaryCrossEntropyLoss final : public LossFunction {
 public:
  BinaryCrossEntropyLoss() {}

  static std::shared_ptr<BinaryCrossEntropyLoss> makeBinaryCrossEntropyLoss() {
    return std::make_shared<BinaryCrossEntropyLoss>();
  }

 private:
  float elementLossGradient(float label, float activation,
                            uint32_t batch_size) const override {
    /* Derivation

    Note: we are assuming that BCE is used along with a signmoid activation in
    the final layer.

    Notation:
     * y - the true label for the neuron in question, y is 0 or 1.
     * a - the activation of the neuron.
     * z - the net inputs of the neuron before applying the activation function.
     * sig - the sigmoid function. Note that d/dx sig(x) = sig(x)(1-sig(x))

    BCE(a,y) = - y log(a) - (1-y) log(1-a)
             = - y log(sig(z)) - (1-y) log(1-sig(z))

    d/dz BCE(a,y)
      = - y [1/sig(x)] sig(x)(1-sig(x)) + (1-y) [1/(1-sig(x))] sig(x)(1-sig(x))
      = - y (1-sig(x)) + (1-y) sig(x)
      = - y + y sig(x) + sig(x) - y sig(x)
      = sig(x) - y

    We are computing y - sig(x) because we want the negative gradient to
    minimize the loss function. We divide by batch size because we average the
    loss over the batch.

    */
    return (label - activation) / batch_size;
  }

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<LossFunction>(this));
  }
};

class MeanSquaredError final : public LossFunction {
 public:
  MeanSquaredError() {}

  static std::shared_ptr<MeanSquaredError> makeMeanSquaredError() {
    return std::make_shared<MeanSquaredError>();
  }

 private:
  float elementLossGradient(float label, float activation,
                            uint32_t batch_size) const override {
    return 2 * (label - activation) / batch_size;
  }

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<LossFunction>(this));
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
  WeightedMeanAbsolutePercentageErrorLoss() {}

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

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<LossFunction>(this));
  }
};

class MarginBCE final : public LossFunction {
 public:
  explicit MarginBCE(float margin) : _margin(margin) {}

  static std::shared_ptr<MarginBCE> makeMarginBCE(float margin) {
    return std::make_shared<MarginBCE>(margin);
  }

 private:
  float elementLossGradient(float label, float activation,
                            uint32_t batch_size) const override {
    if (label == 0.0) {
      activation += _margin;
    } else {
      activation -= _margin;
    }
    return (label - activation) / batch_size;
  }

  float _margin;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<LossFunction>(this), _margin);
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
  if (lower_name == "binarycrossentropyloss" || lower_name == "bce") {
    return BinaryCrossEntropyLoss::makeBinaryCrossEntropyLoss();
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
      "'BinaryCrossEntropyLoss', "
      "'MeanSquaredError'/'MSE', or "
      "'WeightedMeanAbsolutePercentageError'/'WMAPE'");
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::CategoricalCrossEntropyLoss)
CEREAL_REGISTER_TYPE(thirdai::bolt::BinaryCrossEntropyLoss)
CEREAL_REGISTER_TYPE(thirdai::bolt::MeanSquaredError)
CEREAL_REGISTER_TYPE(thirdai::bolt::WeightedMeanAbsolutePercentageErrorLoss)