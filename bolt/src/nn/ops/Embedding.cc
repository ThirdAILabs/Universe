#include "Embedding.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/layers/Optimizer.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt_vector/src/BoltVector.h>
#include <algorithm>
#include <random>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

std::string nextEmbeddingOpName() {
  static uint32_t constructed = 0;
  return "emb" + std::to_string(++constructed);
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

  _embedding_optimizer = AdamOptimizer(_dim * _input_dim);
  _bias_optimizer = AdamOptimizer(_dim);
}

void Embedding::forward(const autograd::ComputationList& inputs,
                        tensor::TensorPtr& output, uint32_t index_in_batch,
                        bool training) {
  (void)training;

  assert(inputs.size() == 2);

  const BoltVector& indices = inputs.at(0)->tensor()->getVector(index_in_batch);
  BoltVector& output_vec = output->getVector(index_in_batch);
  assert(output_vec.isDense());

  if (_bias) {
    std::copy(_biases.begin(), _biases.end(), output_vec.activations);
  } else {
    std::fill_n(output_vec.activations, output_vec.len, 0.F);
  }

  for (size_t n = 0; n < indices.len; n++) {
    uint32_t index = indices.active_neurons[n];
    float value = indices.activations[n];
    const float* emb = _embeddings.data() + index * _dim;

    // #if __GNUC__
    if (n < indices.len - 1) {
      float* nxt_emb =
          _embeddings.data() + indices.active_neurons[n + 1] * _dim;
      __builtin_prefetch(nxt_emb);
    }
    // #endif

#pragma omp simd
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

void Embedding::backpropagate(autograd::ComputationList& inputs,
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

    // #if __GNUC__
    if (n < indices.len - 1) {
      float* nxt_grad = _embedding_optimizer->gradients.data() +
                        indices.active_neurons[n + 1] * _dim;
      __builtin_prefetch(nxt_grad);
    }
    // #endif

#pragma omp simd
    for (size_t i = 0; i < _dim; i++) {
      emb_grad[i] += value * output_vec.gradients[i];
    }
    _embeddings_used[index] = true;

    if (indices.hasGradients()) {
      const float* emb = _embeddings.data() + index * _dim;
      float grad = 0.0;
      for (size_t i = 0; i < _dim; i++) {
        grad += output_vec.gradients[i] * emb[i];
      }
      indices.gradients[n] = grad;
    }
  }

  if (_bias) {
    for (size_t i = 0; i < _dim; i++) {
      _bias_optimizer->gradients[i] += output_vec.gradients[i];
    }
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

void Embedding::updateParameters(float learning_rate, uint32_t train_steps) {
  if (_disable_sparse_parameter_updates) {
    _embedding_optimizer->applyUpdate(_embeddings, learning_rate, train_steps);
  } else {
    sparseEmbeddingUpdate(learning_rate, train_steps);
  }

  if (_bias) {
    _bias_optimizer->applyUpdate(_biases, learning_rate, train_steps);
  }
}

void Embedding::sparseEmbeddingUpdate(float learning_rate,
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
}

void Embedding::summary(std::ostream& summary,
                        const autograd::ComputationList& inputs,
                        const autograd::Computation* output) const {
  summary << "RegularEmbedding(" << name() << "): " << inputs.at(0)->name()
          << " -> " << output->name() << " [dim=" << _dim
          << " input_dim=" << _input_dim
          << " activation=" << activationFunctionToStr(_act_func) << "]";
}

autograd::ComputationPtr Embedding::apply(autograd::ComputationPtr input) {
  if (input->dim() != _input_dim) {
    throw std::invalid_argument(
        "Input has too large of a dimension for embedding.");
  }

  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

template void Embedding::save(cereal::BinaryOutputArchive&) const;

template <class Archive>
void Embedding::save(Archive& archive) const {
  archive(cereal::base_class<Op>(this), _dim, _input_dim, _bias, _act_func,
          _embeddings, _biases, _disable_sparse_parameter_updates,
          _should_serialize_optimizer);

  if (_should_serialize_optimizer) {
    archive(_embedding_optimizer, _bias_optimizer, _embeddings_used);
  }
}

template void Embedding::load(cereal::BinaryInputArchive&);

template <class Archive>
void Embedding::load(Archive& archive) {
  archive(cereal::base_class<Op>(this), _dim, _input_dim, _bias, _act_func,
          _embeddings, _biases, _disable_sparse_parameter_updates,
          _should_serialize_optimizer);

  if (_should_serialize_optimizer) {
    archive(_embedding_optimizer, _bias_optimizer, _embeddings_used);
  } else {
    _embedding_optimizer = AdamOptimizer(_dim * _input_dim);
    _bias_optimizer = AdamOptimizer(_dim);
    _embeddings_used.assign(_input_dim, false);
  }
}

}  // namespace thirdai::bolt::nn::ops

namespace cereal {

/**
 * This is because the Op base class only uses a serialize function, whereas
 * this Op uses a load/save pair. This tells cereal to use the load save pair
 * instead of the serialize method of the parent class. See docs here:
 * https://uscilab.github.io/cereal/serialization_functions.html#inheritance
 */
template <class Archive>
struct specialize<Archive, thirdai::bolt::nn::ops::Embedding,
                  cereal::specialization::member_load_save> {};

}  // namespace cereal

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::Embedding)