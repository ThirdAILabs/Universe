#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <cmath>
#include <stdexcept>

namespace thirdai::bolt {

std::string nextLayerNormOpName() {
  static uint32_t constructed = 0;
  return "layer_norm_" + std::to_string(++constructed);
}

LayerNorm::LayerNorm() : Op(nextLayerNormOpName()) {}

LayerNorm::LayerNorm(const float* gamma, const float* beta, size_t dim)
    : Op(nextLayerNormOpName()),
      _gamma(gamma, gamma + dim),
      _beta(beta, beta + dim),
      _gamma_optimizer(dim),
      _beta_optimizer(dim) {}

std::shared_ptr<LayerNorm> LayerNorm::make() {
  return std::shared_ptr<LayerNorm>(new LayerNorm());
}

std::shared_ptr<LayerNorm> LayerNorm::make(const float* gamma,
                                           const float* beta, size_t dim) {
  return std::shared_ptr<LayerNorm>(new LayerNorm(gamma, beta, dim));
}

void LayerNorm::forward(const ComputationList& inputs, TensorPtr& output,
                        uint32_t index_in_batch, bool training) {
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
  assert(input.len == output.len);

  auto [mean, variance] = moments(input);
  float stddev = std::sqrt(variance + EPSILON);

  if constexpr (!DENSE) {
    std::copy(input.active_neurons, input.active_neurons + input.len,
              output.active_neurons);
  }

  for (uint32_t i = 0; i < input.len; i++) {
    float x_hat = (input.activations[i] - mean) / stddev;

    uint32_t neuron = input.activeNeuronAtIndex<DENSE>(i);

    output.activations[i] = _gamma[neuron] * x_hat + _beta[neuron];
  }
}

void LayerNorm::backpropagate(ComputationList& inputs, TensorPtr& output,
                              uint32_t index_in_batch) {
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
  assert(input.len == output.len);

  // See bolt/src/nn/derivations/LayerNorm.md for the derivation of the code
  // below.

  auto [mean, variance] = moments(input);
  float stddev = std::sqrt(variance + EPSILON);

  float sum_grad = 0.0;
  float sum_grad_x_hat = 0.0;

  for (uint32_t i = 0; i < input.len; i++) {
    float x_hat = (input.activations[i] - mean) / stddev;

    uint32_t neuron = input.activeNeuronAtIndex<DENSE>(i);

    sum_grad += output.gradients[i] * _gamma[neuron];
    sum_grad_x_hat += output.gradients[i] * _gamma[neuron] * x_hat;
  }

  for (uint32_t i = 0; i < input.len; i++) {
    float grad_y = output.gradients[i];
    float x_hat = (input.activations[i] - mean) / stddev;

    uint32_t neuron = input.activeNeuronAtIndex<DENSE>(i);

    float grad_x =
        input.len * grad_y * _gamma[neuron] - sum_grad - sum_grad_x_hat * x_hat;
    grad_x /= (input.len * stddev);

    input.gradients[i] += grad_x;

    _gamma_optimizer.gradients[neuron] += output.gradients[i] * x_hat;

    _beta_optimizer.gradients[neuron] += output.gradients[i];
  }
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
  if (!trainable) {
    return;
  }
  _gamma_optimizer.applyUpdate(_gamma, learning_rate, train_steps);
  _beta_optimizer.applyUpdate(_beta, learning_rate, train_steps);
}

uint32_t LayerNorm::dim() const { return _gamma.size(); }

std::optional<uint32_t> LayerNorm::nonzeros(const ComputationList& inputs,
                                            bool use_sparsity) const {
  return inputs.at(0)->nonzeros(use_sparsity);
}

void LayerNorm::disableSparseParameterUpdates() {}

void LayerNorm::enableSparseParameterUpdates() {}

std::vector<std::vector<float>*> LayerNorm::gradients() {
  return {&_gamma_optimizer.gradients, &_beta_optimizer.gradients};
}

std::vector<std::vector<float>*> LayerNorm::parameters() {
  return {&_gamma, &_beta};
}

void LayerNorm::summary(std::ostream& summary, const ComputationList& inputs,
                        const Computation* output) const {
  summary << "LayerNorm(" << name() << "): " << inputs.at(0)->name() << " -> "
          << output->name();
}

ComputationPtr LayerNorm::apply(const ComputationPtr& input) {
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

  return Computation::make(shared_from_this(), {input});
}

template void LayerNorm::serialize(cereal::BinaryInputArchive&);
template void LayerNorm::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void LayerNorm::serialize(Archive& archive) {
  // The optimizer is small so we can always serialize it.
  archive(cereal::base_class<Op>(this), _gamma, _beta, _gamma_optimizer,
          _beta_optimizer);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::LayerNorm,
                               "thirdai::bolt::nn::ops::LayerNorm")