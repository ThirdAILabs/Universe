#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt_vector/src/BoltVector.h>
#include <cmath>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

// TODO(Nicholas): Fix indexing of scale -> needs to use active_neurons[i] if
// sparse.
void LayerNorm::forward(const autograd::ComputationList& inputs,
                        tensor::TensorPtr& output, uint32_t index_in_batch,
                        bool training) {
  (void)training;

  const BoltVector& input_vector =
      inputs.at(0)->tensor()->getVector(index_in_batch);
  BoltVector& output_vector = output->getVector(index_in_batch);

  auto [mean, variance] = moments(input_vector);

  float stddev = std::sqrt(variance + 1e-7);

  std::copy(input_vector.active_neurons,
            input_vector.active_neurons + input_vector.len,
            output_vector.active_neurons);
  assert(input_vector.len == output_vector.len);
  for (uint32_t i = 0; i < input_vector.len; i++) {
    float centered = (input_vector.activations[i] - mean) / stddev;
    output_vector.activations[i] = _scale[i] * centered + _offset[i];
  }
}

void LayerNorm::backpropagate(autograd::ComputationList& inputs,
                              tensor::TensorPtr& output,
                              uint32_t index_in_batch) {
  /**
   * See bolt/src/nn/ops/derivations/LayerNorm.md for the derivation of this
   * function.
   */

  BoltVector& input_vector = inputs.at(0)->tensor()->getVector(index_in_batch);
  const BoltVector& output_vector = output->getVector(index_in_batch);

  auto [mean, variance] = moments(input_vector);
  float stddev = std::sqrt(variance + 1e-7);

  float partial_wrt_variance =
      partialDerivativeWRTVariance(input_vector, output_vector, mean, stddev);

  float partial_wrt_mean = partialDerivativeWRTMean(
      input_vector, output_vector, mean, stddev, partial_wrt_variance);

  for (uint32_t i = 0; i < input_vector.len; i++) {
    input_vector.gradients[i] += _scale[i] / stddev +
                                 partial_wrt_variance * 2 / input_vector.len *
                                     (input_vector.activations[i] - mean) +
                                 partial_wrt_mean / input_vector.len;

    _scale_optimizer.gradients[i] += output_vector.gradients[i] *
                                     (input_vector.activations[i] - mean) /
                                     stddev;

    _offset_optimizer.gradients[i] += output_vector.gradients[i];
  }
}

float LayerNorm::partialDerivativeWRTVariance(const BoltVector& input_vector,
                                              const BoltVector& output_vector,
                                              float mean, float stddev) {
  float partial_wrt_variance = 0.0;

  for (uint32_t i = 0; i < input_vector.len; i++) {
    float numerator = output_vector.gradients[i] * _scale[i] *
                      (mean - input_vector.activations[i]);
    float denominator = 2 * stddev * stddev * stddev;
    partial_wrt_variance += numerator / denominator;
  }

  return partial_wrt_variance;
}

float LayerNorm::partialDerivativeWRTMean(const BoltVector& input_vector,
                                          const BoltVector& output_vector,
                                          float mean, float stddev,
                                          float partial_wrt_variance) {
  float direct_partial = 0.0;
  float partial_from_variance = 0.0;
  for (uint32_t i = 0; i < input_vector.len; i++) {
    direct_partial += output_vector.gradients[i] * _scale[i];

    partial_from_variance += (input_vector.activations[i] - mean);
  }

  direct_partial /= stddev;

  partial_from_variance *= partial_wrt_variance * 2 / input_vector.len;

  return -direct_partial - partial_from_variance;
}

void LayerNorm::updateParameters(float learning_rate, uint32_t train_steps) {
  _scale_optimizer.applyUpdate(_scale, learning_rate, train_steps);
  _offset_optimizer.applyUpdate(_offset, learning_rate, train_steps);
}

uint32_t LayerNorm::dim() const {
  if (!_dim) {
    throw std::runtime_error(
        "Cannot access dimension of layer norm before applying it to the "
        "result of a computation.");
  }

  return *_dim;
}

std::optional<uint32_t> LayerNorm::nonzeros(
    const autograd::ComputationList& inputs, bool use_sparsity) const {
  return inputs.at(0)->nonzeros(use_sparsity);
}

void LayerNorm::disableSparseParameterUpdates() {}

std::vector<std::vector<float>*> LayerNorm::gradients() const { return {}; }

void LayerNorm::summary(std::ostream& summary,
                        const autograd::ComputationList& inputs,
                        const autograd::Computation* output) const {
  summary << "LayerNorm(" << name() << "): " << inputs.at(0)->name() << " -> "
          << output->name();
}

}  // namespace thirdai::bolt::nn::ops