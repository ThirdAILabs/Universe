#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <cmath>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

std::string nextLayerNormOpName() {
  static uint32_t constructed = 0;
  return "layer_norm_" + std::to_string(++constructed);
}

LayerNorm::LayerNorm() : Op(nextLayerNormOpName()) {}

std::shared_ptr<LayerNorm> LayerNorm::make() {
  return std::shared_ptr<LayerNorm>(new LayerNorm());
}

void LayerNorm::forward(const autograd::ComputationList& inputs,
                        tensor::TensorPtr& output, uint32_t index_in_batch,
                        bool training) {
  (void)training;

  const BoltVector& input_vector =
      inputs.at(0)->tensor()->getVector(index_in_batch);
  BoltVector& output_vector = output->getVector(index_in_batch);

  if (input_vector.isDense()) {
    forward<true>(input_vector, output_vector);
  } else {
    forward<false>(input_vector, output_vector);
  }
}

template <bool DENSE>
void LayerNorm::forward(const BoltVector& input, BoltVector& output) {
  assert(input_vector.len == output_vector.len);

  auto [mean, variance] = moments(input);

  float stddev = std::sqrt(variance + 1e-6);

  std::copy(input.active_neurons, input.active_neurons + input.len,
            output.active_neurons);

  for (uint32_t i = 0; i < input.len; i++) {
    float x_hat = (input.activations[i] - mean) / stddev;

    uint32_t neuron = input.activeNeuronAtIndex<DENSE>(i);

    output.activations[i] = _gamma[neuron] * x_hat + _beta[neuron];
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

  if (input_vector.isDense()) {
    backpropagate<true>(input_vector, output_vector);
  } else {
    backpropagate<false>(input_vector, output_vector);
  }
}

template <bool DENSE>
void LayerNorm::backpropagate(BoltVector& input, const BoltVector& output) {
  auto [mean, variance] = moments(input);
  float stddev = std::sqrt(variance + 1e-6);

  float sum_grad = 0.0;
  float sum_grad_x_hat = 0.0;

  for (uint32_t i = 0; i < input.len; i++) {
    float x_hat = (input.activations[i] - mean) / stddev;

    sum_grad += output.gradients[i];
    sum_grad_x_hat += output.gradients[i] * x_hat;
  }

  for (uint32_t i = 0; i < input.len; i++) {
    float grad_y = output.gradients[i];
    float x_hat = (input.activations[i] - mean) / stddev;

    uint32_t neuron = input.activeNeuronAtIndex<DENSE>(i);

    float grad_x = 0.0;
    grad_x += input.len * grad_y;
    grad_x -= sum_grad;
    grad_x -= sum_grad_x_hat * x_hat;
    grad_x /= (input.len * stddev);

    input.gradients[i] += _gamma[neuron] * grad_x;

    _gamma_optimizer.gradients[neuron] += output.gradients[i] * x_hat;

    _beta_optimizer.gradients[neuron] += output.gradients[i];
  }

  // float grad_variance =
  //     partialDerivativeWRTVariance<DENSE>(input, output, mean, stddev);

  // float grad_mean = partialDerivativeWRTMean<DENSE>(input, output, mean,
  // stddev,
  //                                                   grad_variance);

  // for (uint32_t i = 0; i < input.len; i++) {
  //   float grad_y = output.gradients[i];
  //   float x_hat = (input.activations[i] - mean) / stddev;

  //   uint32_t neuron = input.activeNeuronAtIndex<DENSE>(i);

  //   input.gradients[i] +=
  //       grad_y * _gamma[neuron] / stddev +
  //       grad_variance * 2 / input.len * (output.activations[i] - mean) +
  //       grad_mean / input.len;

  //   _gamma_optimizer.gradients[neuron] += output.gradients[i] * x_hat;

  //   _beta_optimizer.gradients[neuron] += output.gradients[i];
  // }
}

template <bool DENSE>
float LayerNorm::partialDerivativeWRTVariance(const BoltVector& input,
                                              const BoltVector& output,
                                              float mean, float stddev) {
  float partial_wrt_variance = 0.0;

  for (uint32_t i = 0; i < input.len; i++) {
    uint32_t neuron = input.activeNeuronAtIndex<DENSE>(i);

    float numerator =
        output.gradients[i] * _gamma[neuron] * (mean - input.activations[i]);
    float denominator = 2 * stddev * stddev * stddev;
    partial_wrt_variance += numerator / denominator;
  }

  return partial_wrt_variance;
}

template <bool DENSE>
float LayerNorm::partialDerivativeWRTMean(const BoltVector& input,
                                          const BoltVector& output, float mean,
                                          float stddev,
                                          float partial_wrt_variance) {
  float direct_partial = 0.0;
  float partial_from_variance = 0.0;
  for (uint32_t i = 0; i < input.len; i++) {
    uint32_t neuron = input.activeNeuronAtIndex<DENSE>(i);

    direct_partial += output.gradients[i] * _gamma[neuron];

    partial_from_variance += (input.activations[i] - mean);
  }

  direct_partial /= stddev;

  partial_from_variance *= partial_wrt_variance * 2 / input.len;

  return -direct_partial - partial_from_variance;
}

std::pair<float, float> LayerNorm::moments(const BoltVector& vector) {
  float mean = 0.0;
  for (uint32_t i = 0; i < vector.len; i++) {
    mean += vector.activations[i];
  }
  mean /= vector.len;

  float variance = 0.0;
  for (uint32_t i = 0; i < vector.len; i++) {
    float delta = vector.activations[i] - mean;
    variance += (delta * delta);
  }
  variance /= vector.len;

  return {mean, variance};
}

void LayerNorm::updateParameters(float learning_rate, uint32_t train_steps) {
  _gamma_optimizer.applyUpdate(_gamma, learning_rate, train_steps);
  _beta_optimizer.applyUpdate(_beta, learning_rate, train_steps);
}

uint32_t LayerNorm::dim() const { return _gamma.size(); }

std::optional<uint32_t> LayerNorm::nonzeros(
    const autograd::ComputationList& inputs, bool use_sparsity) const {
  return inputs.at(0)->nonzeros(use_sparsity);
}

void LayerNorm::disableSparseParameterUpdates() {}

std::vector<std::vector<float>*> LayerNorm::gradients() {
  return {&_gamma_optimizer.gradients, &_beta_optimizer.gradients};
}

void LayerNorm::summary(std::ostream& summary,
                        const autograd::ComputationList& inputs,
                        const autograd::Computation* output) const {
  summary << "LayerNorm(" << name() << "): " << inputs.at(0)->name() << " -> "
          << output->name();
}

autograd::ComputationPtr LayerNorm::apply(
    const autograd::ComputationPtr& input) {
  if (dim() == 0) {
    _gamma.assign(input->dim(), 1.0);
    _beta.assign(input->dim(), 0.0);
    _gamma_optimizer = AdamOptimizer(input->dim());
    _beta_optimizer = AdamOptimizer(input->dim());
  } else if (input->dim() != dim()) {
    throw std::invalid_argument(
        "Cannot apply LayerNorm op for input with dimension " +
        std::to_string(dim()) + " to input with dimension " +
        std::to_string(input->dim()) + ".");
  }

  return autograd::Computation::make(shared_from_this(), {input});
}

}  // namespace thirdai::bolt::nn::ops