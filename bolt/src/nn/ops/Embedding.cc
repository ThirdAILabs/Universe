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

namespace thirdai::bolt {

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
      _should_serialize_optimizer(false),
      _embeddings_used(input_dim, false) {
  std::mt19937 rng(global_random::nextSeed());
  std::normal_distribution<float> dist(0.0, 0.01);

  auto gen = [&]() { return dist(rng); };
  std::generate(_embeddings.begin(), _embeddings.end(), gen);
  if (_bias) {
    std::generate(_biases.begin(), _biases.end(), gen);
  }
}

void Embedding::forward(const ComputationList& inputs, TensorPtr& output,
                        uint32_t index_in_batch, bool training) {
  (void)training;

  assert(inputs.size() == 1);

  const BoltVector& tokens = inputs.at(0)->tensor()->getVector(index_in_batch);
  BoltVector& output_vec = output->getVector(index_in_batch);
  assert(output_vec.isDense());

  forward(tokens, output_vec.activations);
}

void Embedding::forward(const BoltVector& tokens, float* output) const {
  if (_bias) {
    std::copy(_biases.begin(), _biases.end(), output);
  } else {
    std::fill_n(output, _dim, 0.F);
  }

  for (size_t n = 0; n < tokens.len; n++) {
    float weight = tokens.activations[n];
    const float* emb = embedding(tokens.active_neurons[n]);
    for (size_t i = 0; i < _dim; i++) {
      output[i] += weight * emb[i];
    }
  }

  applyActivationFunction(output);
}

void Embedding::backpropagate(ComputationList& inputs, TensorPtr& output,
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

inline void Embedding::applyActivationFunction(float* activations) const {
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

void Embedding::updateParameters(float learning_rate, uint32_t train_steps) {
  if (_disable_sparse_parameter_updates) {
    _embedding_optimizer->updateDense(_embeddings, _embedding_gradients,
                                      learning_rate, train_steps);
  } else {
    _embedding_optimizer->updateSparseRows(
        _embeddings, _embedding_gradients, _embeddings_used, learning_rate,
        train_steps, /* reset_rows_used= */ true);
  }

  if (_bias) {
    _bias_optimizer->updateDense(_biases, _bias_gradients, learning_rate,
                                 train_steps);
  }
}

void Embedding::initOptimizer(const OptimizerFactoryPtr& optimizer_factory) {
  // The optimizer may be saved (to preserve state in optimizers like Adam)
  // but the gradients are never saved. Thus we only initialize the optimizer
  // if it's not present, but always initialize the gradients, in case we are
  // initializing the optimizer for a loaded model.

  if (!_embedding_optimizer || !_bias_optimizer) {
    _embedding_optimizer = optimizer_factory->makeOptimizer(_input_dim, _dim);
    _bias_optimizer = optimizer_factory->makeOptimizer(/* rows= */ 1, _dim);
  }

  _embedding_gradients.assign(_embeddings.size(), 0.0);
  _bias_gradients.assign(_biases.size(), 0.0);
  _embeddings_used.assign(_input_dim, false);
}

void Embedding::summary(std::ostream& summary, const ComputationList& inputs,
                        const Computation* output) const {
  summary << "Embedding(" << name() << "): " << inputs.at(0)->name() << " -> "
          << output->name() << " [dim=" << _dim
          << ", activation=" << activationFunctionToStr(_act_func)
          << ", bias=" << std::boolalpha << _bias << "]";
}

std::vector<std::pair<std::string, double>> Embedding::parameterAndGradNorms()
    const {
  std::vector<std::pair<std::string, double>> all_norms;

  computeNorms(_embeddings, "embeddings", all_norms);
  if (_embedding_optimizer) {
    computeNorms(_embedding_gradients, "embeddings_grad", all_norms);
  }

  if (_bias) {
    computeNorms(_biases, "bias", all_norms);
    if (_bias_optimizer) {
      computeNorms(_bias_gradients, "bias_grad", all_norms);
    }
  }

  return all_norms;
}

ComputationPtr Embedding::apply(ComputationPtr input) {
  if (input->dim() != _input_dim) {
    throw std::invalid_argument(
        "Input has too large of a dimension for embedding.");
  }

  return Computation::make(shared_from_this(), {std::move(input)});
}

template void Embedding::serialize(cereal::BinaryInputArchive&);
template void Embedding::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Embedding::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this), _dim, _input_dim, _bias, _act_func,
          _embeddings, _biases, _disable_sparse_parameter_updates,
          _should_serialize_optimizer);

  if (_should_serialize_optimizer) {
    archive(_embedding_optimizer, _bias_optimizer);
  }
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::Embedding,
                               "thirdai::bolt::nn::ops::Embedding")
