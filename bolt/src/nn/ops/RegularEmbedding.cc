#include "RegularEmbedding.h"
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/layers/Optimizer.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt_vector/src/BoltVector.h>
#include <algorithm>
#include <random>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

std::string nextRegularEmbeddingOpName() {
  static uint32_t constructed = 0;
  return "reg_emb" + std::to_string(++constructed);
}

RegularEmbedding::RegularEmbedding(size_t dim, size_t input_dim,
                                   const std::string& activation)
    : Op(nextRegularEmbeddingOpName()),
      _dim(dim),
      _input_dim(input_dim),
      _embeddings(dim * input_dim),
      _biases(dim),
      _embeddings_used(input_dim, false),
      _act_func(getActivationFunction(activation)) {
  std::mt19937 rng(global_random::nextSeed());
  std::normal_distribution<float> dist(0.0, 0.01);

  auto gen = [&]() { return dist(rng); };
  std::generate(_embeddings.begin(), _embeddings.end(), gen);
  std::generate(_biases.begin(), _biases.end(), gen);

  _embedding_optimizer = AdamOptimizer(_dim * _input_dim);
  _bias_optimizer = AdamOptimizer(_dim);
}

void RegularEmbedding::forward(const autograd::ComputationList& inputs,
                               tensor::TensorPtr& output,
                               uint32_t index_in_batch, bool training) {
  (void)training;

  assert(inputs.size() == 2);

  const BoltVector& indices = inputs.at(0)->tensor()->getVector(index_in_batch);
  BoltVector& output_vec = output->getVector(index_in_batch);
  assert(output_vec.isDense());

  std::copy(_biases.begin(), _biases.end(), output_vec.activations);

  for (size_t n = 0; n < indices.len; n++) {
    uint32_t index = indices.active_neurons[n];
    float value = indices.activations[n];
    const float* emb = _embeddings.data() + index * _dim;
    for (size_t i = 0; i < _dim; i++) {
      output_vec.activations[i] += value * emb[i];
    }
  }

  float max_act = 0.0;
  for (size_t i = 0; i < _dim; i++) {
    float act = output_vec.activations[i];
    switch (_act_func) {
      case ActivationFunction::ReLU:
        if (act < 0) {
          act = 0;
        }
        break;
      case ActivationFunction::Softmax:
        if (max_act < act) {
          max_act = act;
        }
        break;
      case ActivationFunction::Sigmoid:
        act = 1 / (1 + std::exp(-act));
        break;
      case ActivationFunction::Linear:
        break;
      case ActivationFunction::Tanh:
        act = static_cast<float>(std::tanh(act));
        break;
    }
    output_vec.activations[i] = act;
  }

  if (_act_func == ActivationFunction::Softmax) {
    float total = 0;
    for (size_t n = 0; n < _dim; n++) {
      output_vec.activations[n] = std::exp(output_vec.activations[n] - max_act);
      total += output_vec.activations[n];
    }
    for (size_t n = 0; n < _dim; n++) {
      output_vec.activations[n] /= (total + EPS);
      assert(!std::isnan(output_vec.activations[n]));
    }
  }
}

void RegularEmbedding::backpropagate(autograd::ComputationList& inputs,
                                     tensor::TensorPtr& output,
                                     uint32_t index_in_batch) {
  assert(inputs.size() == 2);

  const BoltVector& indices = inputs.at(0)->tensor()->getVector(index_in_batch);
  BoltVector& output_vec = output->getVector(index_in_batch);
  assert(output_vec.isDense());

  for (size_t i = 0; i < _dim; i++) {
    float act = output_vec.activations[i];
    output_vec.gradients[i] *= actFuncDerivative(act, _act_func);
  }

  for (size_t n = 0; n < indices.len; n++) {
    uint32_t index = indices.active_neurons[n];
    float value = indices.activations[n];
    float* emb_grad = _embedding_optimizer->gradients.data() + index * _dim;
    for (size_t i = 0; i < _dim; i++) {
      emb_grad[i] += value * output_vec.gradients[i];
    }
    _embeddings_used[index] = true;
  }

  for (size_t i = 0; i < _dim; i++) {
    _bias_optimizer->gradients[i] += output_vec.gradients[i];
  }
}

constexpr float momentumUpdate(float curr_momentum, float grad) {
  return BETA1 * curr_momentum + (1 - BETA1) * grad;
}

constexpr float velocityUpdate(float curr_velocity, float grad) {
  return BETA2 * curr_velocity + (1 - BETA2) * grad * grad;
}

constexpr float adam(float momentum, float velocity, float learning_rate,
                     float b1_corrected, float b2_corrected) {
  return learning_rate * (momentum / b1_corrected) /
         (std::sqrt(velocity / b2_corrected) + EPS);
}

void RegularEmbedding::updateParameters(float learning_rate,
                                        uint32_t train_steps) {
  float B1_bias_corrected = static_cast<float>(1 - pow(BETA1, train_steps));
  float B2_bias_corrected = static_cast<float>(1 - pow(BETA2, train_steps));

#pragma omp parallel for default(none) \
    shared(B1_bias_corrected, B2_bias_corrected, learning_rate)
  for (size_t n = 0; n < _input_dim; n++) {
    if (!_embeddings_used[n]) {
      continue;
    }
    _embeddings_used[n] = false;
    for (size_t i = 0; i < _dim; i++) {
      size_t index = n * _dim + i;
      float grad = _embedding_optimizer->gradients[index];

      _embedding_optimizer->momentum[index] =
          momentumUpdate(_embedding_optimizer->momentum[index], grad);

      _embedding_optimizer->velocity[index] =
          velocityUpdate(_embedding_optimizer->velocity[index], grad);

      assert(!std::isnan(_embedding_optimizer->momentum[index]));
      assert(!std::isnan(_embedding_optimizer->velocity[index]));

      _embeddings[index] +=
          adam(_embedding_optimizer->momentum[index],
               _embedding_optimizer->velocity[index], learning_rate,
               B1_bias_corrected, B2_bias_corrected);

      assert(!std::isnan(_embeddings[index]));

      _embedding_optimizer->gradients[index] = 0;
    }
  }

  for (size_t i = 0; i < _dim; i++) {
    float grad = _bias_optimizer->gradients[i];

    _bias_optimizer->momentum[i] =
        momentumUpdate(_bias_optimizer->momentum[i], grad);

    _bias_optimizer->velocity[i] =
        velocityUpdate(_bias_optimizer->velocity[i], grad);

    assert(!std::isnan(_bias_optimizer->momentum[i]));
    assert(!std::isnan(_bias_optimizer->velocity[i]));

    _biases[i] +=
        adam(_bias_optimizer->momentum[i], _bias_optimizer->velocity[i],
             learning_rate, B1_bias_corrected, B2_bias_corrected);
    assert(!std::isnan(_biases[i]));

    _bias_optimizer->gradients[i] = 0;
  }
}

void RegularEmbedding::summary(std::ostream& summary,
                               const autograd::ComputationList& inputs,
                               const autograd::Computation* output) const {
  summary << "RegularEmbedding(" << name() << "): " << inputs.at(0)->name()
          << " -> " << output->name() << " [dim=" << _dim
          << " input_dim=" << _input_dim
          << " activation=" << activationFunctionToStr(_act_func) << "]";
}

autograd::ComputationPtr RegularEmbedding::apply(
    autograd::ComputationPtr input) {
  if (input->dim() != _input_dim) {
    throw std::invalid_argument(
        "Input has too large of a dimension for embedding.");
  }

  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

}  // namespace thirdai::bolt::nn::ops