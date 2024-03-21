#include "QuantileMixing.h"
#include <bolt/src/nn/autograd/Computation.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <algorithm>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>

namespace thirdai::bolt {

std::string nextQuantileMixingOpName() {
  static uint32_t constructed = 0;
  return "max_pool_1d_" + std::to_string(++constructed);
}

QuantileMixing::QuantileMixing(size_t window_size, float frac)
    : Op(nextQuantileMixingOpName()), _window_size(window_size), _frac(frac) {
  if (frac <= 0.0 || frac >= 1.0) {
    throw std::invalid_argument(
        "Frac argument to QuantilMixing must be in the range (0,1).");
  }
}

void QuantileMixing::forward(const ComputationList& inputs, TensorPtr& output,
                             uint32_t index_in_batch, bool training) {
  (void)training;
  assert(inputs.size() == 1);

  const BoltVector& input_vec =
      inputs.at(0)->tensor()->getVector(index_in_batch);
  assert(input_vec.isDense());
  assert(input_vec.len % _window_size == 0);

  BoltVector& output_vec = output->getVector(index_in_batch);
  assert(output_vec.isDense());
  assert(output_vec.len == input_vec.len);

  size_t nth = (1 - _frac) * _window_size;

  std::vector<float> window(_window_size);
  for (size_t start = 0; start < output_vec.len; start += _window_size) {
    const float* input_ptr = input_vec.activations + start;
    float* output_ptr = output_vec.activations + start;

    // Copy because nth_element modifies the array.
    std::copy(input_ptr, input_ptr + _window_size, window.begin());
    std::nth_element(window.begin(), window.begin() + nth, window.end());
    float threshold = window[nth];

    for (size_t i = 0; i < _window_size; i++) {
      if (input_ptr[i] >= threshold) {
        output_ptr[i] = 1;
      } else {
        output_ptr[i] = 0;
      }
    }
  }
}

void QuantileMixing::backpropagate(ComputationList& inputs, TensorPtr& output,
                                   uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  BoltVector& input_vec = inputs.at(0)->tensor()->getVector(index_in_batch);
  assert(input_vec.isDense());
  assert(input_vec.len % _window_size == 0);

  if (!input_vec.hasGradients()) {
    return;
  }

  const BoltVector& output_vec = output->getVector(index_in_batch);
  assert(output_vec.isDense());
  assert(output_vec.len == input_vec.len);

  size_t nth = (1 - _frac) * _window_size;

  std::vector<float> window(_window_size);
  for (size_t start = 0; start < output_vec.len; start += _window_size) {
    const float* input_ptr = input_vec.activations + start;
    float* input_grads = input_vec.gradients + start;
    const float* output_grads = output_vec.gradients + start;

    // Copy because nth_element modifies the array.
    std::copy(input_ptr, input_ptr + _window_size, window.begin());
    std::nth_element(window.begin(), window.begin() + nth, window.end());
    float threshold = window[nth];

    for (size_t i = 0; i < _window_size; i++) {
      if (input_ptr[i] >= threshold) {
        input_grads[i] = output_grads[i];
      } else {
        input_grads[i] = 0;
      }
    }
  }
}

void QuantileMixing::updateParameters(float learning_rate,
                                      uint32_t train_steps) {
  (void)learning_rate;
  (void)train_steps;
}

uint32_t QuantileMixing::dim() const { return _output_dim; }

std::optional<uint32_t> QuantileMixing::nonzeros(const ComputationList& inputs,
                                                 bool use_sparsity) const {
  (void)inputs;
  (void)use_sparsity;
  return _output_dim;
}

void QuantileMixing::initOptimizer(const OptimizerFactoryPtr& optimizer_factory,
                                   bool replace_existing_optimizer) {
  (void)optimizer_factory;
  (void)replace_existing_optimizer;
}

void QuantileMixing::disableSparseParameterUpdates() {}

void QuantileMixing::enableSparseParameterUpdates() {}

std::vector<std::vector<float>*> QuantileMixing::gradients() { return {}; }

std::vector<std::vector<float>*> QuantileMixing::parameters() { return {}; }

ComputationPtr QuantileMixing::applyToInputs(const ComputationList& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("QuantileMixing op expects a single input.");
  }
  return apply(inputs.at(0));
}

ar::ConstArchivePtr QuantileMixing::toArchive(bool with_optimizer) const {
  (void)with_optimizer;

  auto map = baseArchive();

  map->set("type", ar::str(type()));
  map->set("output_dim", ar::u64(_output_dim));
  map->set("window_size", ar::u64(_window_size));
  map->set("frac", ar::f32(_frac));

  return map;
}

std::shared_ptr<QuantileMixing> QuantileMixing::fromArchive(
    const ar::Archive& archive) {
  return std::shared_ptr<QuantileMixing>(new QuantileMixing(archive));
}

QuantileMixing::QuantileMixing(const ar::Archive& archive)
    : Op(archive.str("name")),
      _output_dim(archive.u64("output_dim")),
      _window_size(archive.u64("window_size")),
      _frac(archive.getAs<ar::F32>("frac")) {
  assertOpType(archive, type());
}

void QuantileMixing::summary(std::ostream& summary,
                             const ComputationList& inputs,
                             const Computation* output) const {
  summary << "QuantileMixing(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name() << "[window_size=" << _window_size
          << ", frac=" << _frac << "]";
}

void QuantileMixing::setSerializeOptimizer(bool should_serialize_optimizer) {
  (void)should_serialize_optimizer;
}

ComputationPtr QuantileMixing::apply(ComputationPtr input) {
  if (input->dim() == 0) {
    throw std::invalid_argument(
        "Cannot apply quantile mixing to input with dim=0.");
  }
  if (_output_dim == 0) {  // 0 means unset since output dim cannot be 0.
    _output_dim = input->dim();
  }
  if (_output_dim % _window_size != 0) {
    throw std::invalid_argument(
        "Output dim must be a multiple of window size for QuantileMixing.");
  }
  if (input->dim() != _output_dim) {
    throw std::invalid_argument("Cannot apply QuantileMixing with output dim " +
                                std::to_string(_output_dim) +
                                " to input with dim " +
                                std::to_string(input->dim()) + ".");
  }

  return Computation::make(shared_from_this(), {std::move(input)});
}

}  // namespace thirdai::bolt