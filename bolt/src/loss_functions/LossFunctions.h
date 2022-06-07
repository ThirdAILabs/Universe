#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/metrics/Metric.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt {

class LossFunction : public Metric {
 public:
  LossFunction() : _loss(0.0), _num_samples(0) {}

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

  void computeMetric(const BoltVector& output, const BoltVector& labels) final {
    if (output.isDense()) {
      if (labels.isDense()) {
        computeLossImpl<true, true>(output, labels);
      } else {
        computeLossImpl<true, false>(output, labels);
      }
    } else {
      if (labels.isDense()) {
        throw std::invalid_argument(
            "Having sparse output and dense labels is not supported.");
      }
      computeLossImpl<false, false>(output, labels);
    }
  }

  double getMetricAndReset(bool verbose) final {
    double loss = _loss.load(std::memory_order_relaxed) / _num_samples;
    if (verbose) {
      std::cout << "Loss: " << loss << std::endl;
    }
    _loss = 0.0;
    _num_samples = 0;
    return loss;
  }

  bool forceDenseInference() final { return true; }

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

  template <bool OUTPUT_DENSE, bool LABEL_DENSE>
  void computeLossImpl(const BoltVector& output, const BoltVector& labels) {
    // We cannot have sparse output and dense labels. This will cause errors in
    // sampling and means that there is no valid definition of CrossEntropy loss
    // since non-active neurons have 0 activation and log(0) is undefined.
    static_assert(OUTPUT_DENSE || !LABEL_DENSE);
    if (OUTPUT_DENSE) {
      // If either of the the vectors is dense then we have to iterate over the
      // full dimension. To find this dimension we can take the max of the
      // dimensions of both vectors since we know that at least one is dense.

      float sample_loss = 0.0;
      for (uint32_t i = 0; i < output.len; i++) {
        float activation = output.findActiveNeuron<OUTPUT_DENSE>(i).activation;
        float label_val = labels.findActiveNeuron<LABEL_DENSE>(i).activation;

        sample_loss += elementLoss(label_val, activation);
      }

      MetricUtilities::incrementAtomicFloat(_loss, sample_loss);
      _num_samples += 1;
    } else {
      // Since we cannot have sparse outputs, and dense labels, in this case we
      // know that both the outputs and labels are sparse. If both vectors are
      // sparse then we can only iterate over the nonzero indices of the
      // outputs. This is because we currently support loss metrics during
      // training, and thus we know that the labels are a subset of the active
      // neurons. Furthermore this is an important distinction because the
      // activation of a non-active neuron is defined as 0, which means that
      // both binary and categorical cross entropy are not well defined since
      // they involve computing the log of the activation. Thus we can only
      // compute the loss for neurons that are active because that ensures that
      // the activation is non-zero.

      float sample_loss = 0.0;
      for (uint32_t i = 0; i < output.len; i++) {
        float label_val =
            labels.findActiveNeuron<LABEL_DENSE>(output.active_neurons[i])
                .activation;
        if (label_val == 0.0) {
          sample_loss += elementLoss(label_val, output.activations[i]);
        }
      }

      MetricUtilities::incrementAtomicFloat(_loss, sample_loss);
      _num_samples += 1;
    }
  }

  virtual float elementLoss(float label, float activation) const = 0;

  std::atomic<float> _loss;
  std::atomic<uint32_t> _num_samples;
};

class CategoricalCrossEntropyLoss final : public LossFunction {
 public:
  static std::shared_ptr<CategoricalCrossEntropyLoss>
  makeCategoricalCrossEntropyLoss() {
    return std::make_shared<CategoricalCrossEntropyLoss>();
  }

  std::string getName() final { return "categorical_cross_entropy_loss"; }

 private:
  float elementLossGradient(float label, float activation,
                            uint32_t batch_size) const final {
    return (label - activation) / batch_size;
  }

  float elementLoss(float label, float activation) const final {
    /*
      CrossEntropyLoss is defined as Loss = -∑ y_i log(a_i)
      where y_i is the ith label and a_i is the ith activation.

      Thus we will each element as label * log(activation).
    */
    return -label * log(activation);
  }
};

class BinaryCrossEntropyLoss final : public LossFunction {
 public:
  static std::shared_ptr<BinaryCrossEntropyLoss> makeBinaryCrossEntropyLoss() {
    return std::make_shared<BinaryCrossEntropyLoss>();
  }

  std::string getName() final { return "binary_cross_entropy_loss"; }

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

  float elementLoss(float label, float activation) const final {
    /*
      BinaryCrossEntropyLoss is is a special case of CrossEntropyLoss for 2
      classes. It is defined as Loss = -[y log(a) + (1-y)log(1-a)] where y is
      the label and a is the activation. Since we are treating each element as
      its own class we will use this formula for each element.
    */
    float log_act = log(activation);
    return -(label * log_act + (1 - label) * (1 - log_act));
  }
};

class MeanSquaredError final : public LossFunction {
 public:
  static std::shared_ptr<MeanSquaredError> makeMeanSquaredError() {
    return std::make_shared<MeanSquaredError>();
  }

  std::string getName() final { return "mean_squared_error"; }

 private:
  float elementLossGradient(float label, float activation,
                            uint32_t batch_size) const override {
    return 2 * (label - activation) / batch_size;
  }

  float elementLoss(float label, float activation) const final {
    /*
      MeanSquaredError is defined as Error = ∑ (y_i - a_i)^2
      where y_i is the ith label and a_i is the ith activation.

      Thus we will each element as (y_i - a_i)^2.
    */
    float diff = label - activation;
    return diff * diff;
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

  std::string getName() final {
    return "weighted_mean_absolute_percentage_error";
  }

 private:
  float elementLossGradient(float label, float activation,
                            uint32_t batch_size) const override {
    auto direction = activation > label ? -1.0 : 1.0;
    return direction / batch_size;
  }

  float elementLoss(float label, float activation) const final {
    // WMAPE has a special way that the loss metric is computed so we will defer
    // to using that metric instead of computing it as part of the loss fuction.
    (void)label;
    (void)activation;
    throw std::runtime_error(
        "Please use specific weighted_mean_absolute_percentage_error metric "
        "instead of just specifying loss.");
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
