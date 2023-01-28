#pragma once

#include <cereal/types/polymorphic.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <utils/StringManipulation.h>
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt {

class LossFunction {
 public:
  LossFunction() {}

  void lossGradients(BoltVector& output, const BoltVector& labels,
                     uint32_t batch_size) const;

  virtual ~LossFunction() = default;

 private:
  template <bool OUTPUT_DENSE, bool LABEL_DENSE>
  void computeLossGradientsImpl(BoltVector& output, const BoltVector& labels,
                                uint32_t batch_size) const;

  virtual float elementLossGradient(float label, float activation,
                                    uint32_t batch_size) const = 0;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
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
  void serialize(Archive& archive);
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
  void serialize(Archive& archive);
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
  void serialize(Archive& archive);
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
  void serialize(Archive& archive);
};

class MarginBCE final : public LossFunction {
 public:
  MarginBCE(float positive_margin, float negative_margin, bool bound)
      : _positive_margin(positive_margin),
        _negative_margin(negative_margin),
        _bound(bound) {}

 private:
  // Private constructor for cereal
  MarginBCE() {}
  float elementLossGradient(float label, float activation,
                            uint32_t batch_size) const override {
    if (label == 0.0) {
      activation += _negative_margin;
    } else {
      activation -= _positive_margin;
    }
    if (_bound) {
      activation = std::min<float>(activation, 1.0);
      activation = std::max<float>(activation, 0.0);
    }
    return (label - activation) / batch_size;
  }

  float _positive_margin;
  float _negative_margin;
  bool _bound;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

static std::shared_ptr<LossFunction> getLossFunction(const std::string& name) {
  std::string lower_name = text::lower(name);
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
