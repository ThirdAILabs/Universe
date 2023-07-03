#include "Embedding.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/utils/Timer.h>
#include <bolt_vector/src/BoltVector.h>
#include <algorithm>
#include <chrono>
#include <ios>
#include <random>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

std::string nextEmbeddingOpName() {
  static uint32_t constructed = 0;
  return "emb_" + std::to_string(++constructed);
}

Embedding::Embedding(size_t dim, size_t input_dim,
                     const std::string& activation, bool bias)
    : Op(nextEmbeddingOpName()),
      _dim(dim),
      _input_dim(input_dim),
      _bias(bias),
      _act_func(getActivationFunction(activation)),
      _embeddings(dim * input_dim),
      _biases(dim, 0.0),
      _disable_sparse_parameter_updates(false),
      _embeddings_used(input_dim, false) {
  std::mt19937 rng(global_random::nextSeed());
  std::normal_distribution<float> dist(0.0, 0.01);

  auto gen = [&]() { return dist(rng); };
  std::generate(_embeddings.begin(), _embeddings.end(), gen);
  if (_bias) {
    std::generate(_biases.begin(), _biases.end(), gen);
  }
}

void Embedding::forward(const autograd::ComputationList& inputs,
                        tensor::TensorPtr& output, uint32_t index_in_batch,
                        bool training) {
  (void)training;

  assert(inputs.size() == 1);

  const BoltVector& tokens = inputs.at(0)->tensor()->getVector(index_in_batch);
  BoltVector& output_vec = output->getVector(index_in_batch);
  assert(output_vec.isDense());

  if (_bias) {
    std::copy(_biases.begin(), _biases.end(), output_vec.activations);
  } else {
    std::fill_n(output_vec.activations, output_vec.len, 0.F);
  }

  for (size_t n = 0; n < tokens.len; n++) {
    float weight = tokens.activations[n];
    const float* emb = embedding(tokens.active_neurons[n]);
    for (size_t i = 0; i < _dim; i++) {
      output_vec.activations[i] += weight * emb[i];
    }
  }

  applyActivationFunction(output_vec.activations);
}

void Embedding::backpropagate(autograd::ComputationList& inputs,
                              tensor::TensorPtr& output,
                              uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  const BoltVector& tokens = inputs.at(0)->tensor()->getVector(index_in_batch);
  BoltVector& output_vec = output->getVector(index_in_batch);
  assert(output_vec.isDense());

  applyActivationFunctionGrad(output_vec.activations, output_vec.gradients);

  for (size_t n = 0; n < tokens.len; n++) {
    uint32_t token = tokens.active_neurons[n];
    float weight = tokens.activations[n];
    float* emb_grad = gradients(token);
    for (size_t i = 0; i < _dim; i++) {
      emb_grad[i] += weight * output_vec.gradients[i];
    }
    _embeddings_used[token] = true;

    if (tokens.hasGradients()) {
      const float* emb = embedding(token);
      float weight_grad = 0.0;
      for (size_t i = 0; i < _dim; i++) {
        weight_grad += output_vec.gradients[i] * emb[i];
      }
      tokens.gradients[n] = weight_grad;
    }
  }

  if (_bias) {
    for (size_t i = 0; i < _dim; i++) {
      _bias_gradients[i] += output_vec.gradients[i];
    }
  }
}

void softmax(float* activations, size_t dim) {
  float max_act = 0.0;
  for (size_t i = 0; i < dim; i++) {
    float act = activations[i];
    if (max_act < act) {
      max_act = act;
    }
  }

  float total = 0;
  for (size_t i = 0; i < dim; i++) {
    float act = activations[i];
    act = std::exp(act - max_act);
    activations[i] = act;
    total += act;
  }
  for (size_t i = 0; i < dim; i++) {
    activations[i] /= (total + EPS);
    assert(!std::isnan(activations[i]));
  }
}

void Embedding::applyActivationFunction(float* activations) {
  switch (_act_func) {
    case ActivationFunction::ReLU:
      for (size_t i = 0; i < _dim; i++) {
        if (activations[i] < 0) {
          activations[i] = 0;
        }
      }
      break;
    case ActivationFunction::Softmax:
      softmax(activations, _dim);
      break;
    case ActivationFunction::Sigmoid:
      for (size_t i = 0; i < _dim; i++) {
        activations[i] = 1 / (1 + std::exp(-activations[i]));
      }
      break;
    case ActivationFunction::Linear:
      break;
    case ActivationFunction::Tanh:
      for (size_t i = 0; i < _dim; i++) {
        activations[i] = std::tanh(activations[i]);
      }
      break;
  }
}

void Embedding::applyActivationFunctionGrad(const float* activations,
                                            float* gradients) {
  for (size_t i = 0; i < _dim; i++) {
    float act = activations[i];
    gradients[i] *= actFuncDerivative(act, _act_func);
  }
}

int64_t total_update_time = 0;
int64_t sparse_update_time_outside = 0;

void Embedding::updateParameters(float learning_rate, uint32_t train_steps) {
  utils::Timer timer;

  if (_disable_sparse_parameter_updates) {
    _embedding_optimizer->updateDense(_embeddings, _embedding_gradients,
                                      learning_rate, train_steps);
  } else {
    utils::Timer t1;

    _embedding_optimizer->updateSparseRows(
        _embeddings, _embedding_gradients, _embeddings_used, learning_rate,
        train_steps, /* reset_rows_used= */ true);

    t1.stop();
    sparse_update_time_outside += t1.elapsed<std::chrono::milliseconds>();
  }

  if (_bias) {
    _bias_optimizer->updateDense(_biases, _bias_gradients, learning_rate,
                                 train_steps);
  }

  timer.stop();

  total_update_time += timer.elapsed<std::chrono::milliseconds>();
}

Embedding::~Embedding() {
  std::cerr << "Optimizers: total_update_time=" << total_update_time
            << std::endl;

  std::cerr << "sparse_update_time_outside_opt=" << sparse_update_time_outside
            << std::endl;
}

void Embedding::initOptimizer(const optimizers::Factory& optimizer_factory) {
  if (!_embedding_optimizer || !_bias_optimizer) {
    _embedding_optimizer = optimizer_factory.makeOptimizer(_input_dim, _dim);
    _bias_optimizer = optimizer_factory.makeOptimizer(/* rows= */ 1, _dim);

    _embedding_gradients.assign(_embeddings.size(), 0.0);
    _bias_gradients.assign(_biases.size(), 0.0);
  }
}

void Embedding::summary(std::ostream& summary,
                        const autograd::ComputationList& inputs,
                        const autograd::Computation* output) const {
  summary << "Embedding(" << name() << "): " << inputs.at(0)->name() << " -> "
          << output->name() << " [dim=" << _dim
          << ", activation=" << activationFunctionToStr(_act_func)
          << ", bias=" << std::boolalpha << _bias << "]";
}

autograd::ComputationPtr Embedding::apply(autograd::ComputationPtr input) {
  if (input->dim() != _input_dim) {
    throw std::invalid_argument(
        "Input has too large of a dimension for embedding.");
  }

  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

template void Embedding::serialize(cereal::BinaryInputArchive&);
template void Embedding::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Embedding::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this), _dim, _input_dim, _bias, _act_func,
          _embeddings, _biases, _disable_sparse_parameter_updates,
          _embedding_optimizer, _bias_optimizer);

  // We never save the gradients from a particular batch. Users should call
  // updateParameters to apply an update before saving, or process the training
  // batch again after loading. This will ensure they are properly initialized
  // when loading.
  if (_embedding_gradients.empty()) {
    _embedding_gradients.assign(_embeddings.size(), 0.0);
  }
  if (_bias_gradients.empty()) {
    _bias_gradients.assign(_biases.size(), 0.0);
  }
  if (_embeddings_used.empty()) {
    _embeddings_used.assign(_input_dim, false);
  }
}

}  // namespace thirdai::bolt::nn::ops

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::Embedding)