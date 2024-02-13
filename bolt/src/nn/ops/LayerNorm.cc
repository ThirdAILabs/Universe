#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <archive/src/ParameterReference.h>
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
      _beta(beta, beta + dim) {}

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

    _gamma_gradients[neuron] += output.gradients[i] * x_hat;

    _beta_gradients[neuron] += output.gradients[i];
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
  _gamma_optimizer->updateDense(_gamma, _gamma_gradients, learning_rate,
                                train_steps);
  _beta_optimizer->updateDense(_beta, _beta_gradients, learning_rate,
                               train_steps);
}

void LayerNorm::initOptimizer(const OptimizerFactoryPtr& optimizer_factory) {
  // The optimizer may be saved (to preserve state in optimizers like Adam)
  // but the gradients are never saved. Thus we only initialize the optimizer
  // if it's not present, but always initialize the gradients, in case we are
  // initializing the optimizer for a loaded model.

  if (!_gamma_optimizer || !_beta_optimizer) {
    _gamma_optimizer =
        optimizer_factory->makeOptimizer(/* rows= */ 1, _gamma.size());
    _beta_optimizer =
        optimizer_factory->makeOptimizer(/* rows= */ 1, _beta.size());
  }

  _gamma_gradients.assign(_gamma.size(), 0.0);
  _beta_gradients.assign(_beta.size(), 0.0);
}

uint32_t LayerNorm::dim() const { return _gamma.size(); }

std::optional<uint32_t> LayerNorm::nonzeros(const ComputationList& inputs,
                                            bool use_sparsity) const {
  return inputs.at(0)->nonzeros(use_sparsity);
}

ComputationPtr LayerNorm::applyToInputs(const ComputationList& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("Expected LayerNorm op to have one input.");
  }
  return apply(inputs.at(0));
}

ar::ConstArchivePtr LayerNorm::toArchive(bool with_optimizer) const {
  auto map = baseArchive();
  map->set("type", ar::str(type()));

  map->set("gamma", ar::ParameterReference::make(_gamma, shared_from_this()));
  map->set("beta", ar::ParameterReference::make(_beta, shared_from_this()));

  if (with_optimizer) {
    map->set("gamma_optimizer",
             _gamma_optimizer->toArchive(shared_from_this()));

    map->set("beta_optimizer", _beta_optimizer->toArchive(shared_from_this()));
  }

  return map;
}

std::shared_ptr<LayerNorm> LayerNorm::fromArchive(const ar::Archive& archive) {
  return std::shared_ptr<LayerNorm>(new LayerNorm(archive));
}

LayerNorm::LayerNorm(const ar::Archive& archive)
    : Op(archive.str("name")),
      _gamma(archive.get("gamma")->param().moveLoadedParameter()),
      _beta(archive.get("beta")->param().moveLoadedParameter()) {
  assertOpType(archive, type());

  if (archive.contains("gamma_optimizer")) {
    _gamma_optimizer = Optimizer::fromArchive(*archive.get("gamma_optimizer"));
  }
  if (archive.contains("beta_optimizer")) {
    _beta_optimizer = Optimizer::fromArchive(*archive.get("beta_optimizer"));
  }
}

void LayerNorm::disableSparseParameterUpdates() {}

void LayerNorm::enableSparseParameterUpdates() {}

std::vector<std::vector<float>*> LayerNorm::gradients() {
  return {&_gamma_gradients, &_beta_gradients};
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
  archive(cereal::base_class<Op>(this), _gamma, _beta, _gamma_gradients,
          _beta_gradients, _gamma_optimizer, _beta_optimizer);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::LayerNorm,
                               "thirdai::bolt::nn::ops::LayerNorm")