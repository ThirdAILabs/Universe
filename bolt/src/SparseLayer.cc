#include "SparseLayer.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>

namespace thirdai::bolt {

SparseLayer::SparseLayer(uint64_t _dim, uint64_t _prev_dim, float _sparsity,
                         ActivationFunc _act_func,
                         SamplingConfig _sampling_config)
    : dim(_dim),
      prev_dim(_prev_dim),
      batch_size(0),
      sparse_dim(_sparsity * _dim),
      sparsity(_sparsity),
      act_func(_act_func),
      active_lens(nullptr),
      active_neurons(nullptr),
      activations(nullptr),
      errors(nullptr),
      sampling_config(_sampling_config) {
  uint64_t total_size = dim * prev_dim;

  weights = new float[total_size];
  w_gradient = new float[total_size]();
  w_momentum = new float[total_size]();
  w_velocity = new float[total_size]();

  biases = new float[dim];
  b_gradient = new float[dim]();
  b_momentum = new float[dim]();
  b_velocity = new float[dim]();

  is_active = new bool[dim]();  // TODO(nicholas): bitvector?

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::normal_distribution<float> dist(0.0, 0.01);

  std::generate(weights, weights + total_size, [&]() { return dist(eng); });
  std::generate(biases, biases + dim, [&]() { return dist(eng); });

  if (sparsity < 1.0) {
    hasher = new utils::DWTAHashFunction(
        prev_dim, sampling_config.hashes_per_table, sampling_config.num_tables,
        sampling_config.range_pow);

    hash_table = new utils::SampledHashTable<uint32_t>(
        sampling_config.num_tables, sampling_config.reservoir_size,
        sampling_config.range_pow);

    BuildHashTables();

    rand_neurons = new uint32_t[dim];
    for (uint32_t i = 0; i < dim; i++) {
      rand_neurons[i] = i;
    }

    std::shuffle(rand_neurons, rand_neurons + dim, rd);
  } else {
    hasher = nullptr;
    hash_table = nullptr;
    rand_neurons = nullptr;
  }
}

void SparseLayer::FeedForward(uint32_t batch_indx, const uint32_t* indices,
                              const float* values, uint32_t len,
                              uint32_t* labels, uint32_t label_len) {
  SelectActiveNeurons(batch_indx, indices, values, len, labels, label_len);

  float max_act = 0;
  for (uint64_t n = 0; n < active_lens[batch_indx]; n++) {
    uint64_t act_neuron = active_neurons[batch_indx][n];
    is_active[act_neuron] = true;
    float act = biases[act_neuron];
    for (uint64_t i = 0; i < len; i++) {
      act += weights[act_neuron * prev_dim + indices[i]] * values[i];
    }
    switch (act_func) {
      case ActivationFunc::ReLU:
        if (act < 0) {
          activations[batch_indx][n] = 0;
        } else {
          activations[batch_indx][n] = act;
        }
        break;
      case ActivationFunc::Softmax:
        activations[batch_indx][n] = act;
        if (max_act < act) {
          max_act = act;
        }
        break;
    }
  }

  if (act_func == ActivationFunc::Softmax) {
    float total = 0;
    for (uint64_t n = 0; n < active_lens[batch_indx]; n++) {
      activations[batch_indx][n] =
          std::exp(activations[batch_indx][n] - max_act);
      total += activations[batch_indx][n];
    }
    for (uint64_t n = 0; n < active_lens[batch_indx]; n++) {
      activations[batch_indx][n] /= (total + EPS);
    }
  }
}

constexpr float SparseLayer::ActFuncDerivative(float x) {
  switch (act_func) {
    case ActivationFunc::ReLU:
      return x > 0 ? 1.0 : 0.0;
    case ActivationFunc::Softmax:
      return 1.0;
  }
  // This is impossible to reach, but the compiler gave a warning saying it
  // reached the end of a non void function wihtout it.
  return 0.0;
}

template void SparseLayer::BackPropagateImpl<true>(uint32_t, const uint32_t*,
                                               const float*, float*, uint32_t);

template void SparseLayer::BackPropagateImpl<false>(uint32_t, const uint32_t*,
                                                const float*, float*, uint32_t);

template <bool FIRST_LAYER>
void SparseLayer::BackPropagateImpl(uint32_t batch_indx, const uint32_t* indices,
                                const float* values, float* prev_errors,
                                uint32_t len) {
  for (uint64_t n = 0; n < active_lens[batch_indx]; n++) {
    errors[batch_indx][n] *= ActFuncDerivative(activations[batch_indx][n]);
    for (uint64_t i = 0; i < len; i++) {
      w_gradient[active_neurons[batch_indx][n] * prev_dim + indices[i]] +=
          errors[batch_indx][n] * values[i];
      if (!FIRST_LAYER) {
        prev_errors[i] +=
            errors[batch_indx][n] *
            weights[active_neurons[batch_indx][n] * prev_dim + indices[i]];
      }
    }
    b_gradient[active_neurons[batch_indx][n]] += errors[batch_indx][n];
  }
}

void SparseLayer::ComputeErrors(uint32_t batch_indx, const uint32_t* labels,
                                uint32_t label_len) {
  float frac = 1.0 / label_len;

  for (uint64_t n = 0; n < active_lens[batch_indx]; n++) {
    if (std::find(labels, labels + label_len, active_neurons[batch_indx][n]) !=
        labels + label_len) {
      errors[batch_indx][n] = (frac - activations[batch_indx][n]) / batch_size;
    } else {
      errors[batch_indx][n] = -activations[batch_indx][n] / batch_size;
    }
  }
}

void SparseLayer::SelectActiveNeurons(uint32_t batch_indx,
                                      const uint32_t* indices,
                                      const float* values, uint32_t len,
                                      uint32_t* labels, uint32_t label_len) {
  if (sparsity == 1.0) {
    active_lens[batch_indx] = dim;
    for (uint32_t i = 0; i < dim; i++) {
      active_neurons[batch_indx][i] = i;
    }
  } else {
    std::unordered_set<uint32_t> active_set;

    for (uint32_t i = 0; i < label_len; i++) {
      active_set.insert(labels[i]);
    }

    uint32_t* hashes = new uint32_t[hash_table->numTables()];
    hasher->hashSingleSparse(indices, values, len, hashes);
    hash_table->queryBySet(hashes, active_set);
    delete[] hashes;

    if (active_set.size() < sparse_dim) {
      uint32_t rand_offset = rand() % dim;
      while (active_set.size() < sparse_dim) {
        active_set.insert(rand_neurons[rand_offset++]);
        rand_offset = rand_offset % dim;
      }
    }

    uint32_t active_len = sparse_dim;
    active_lens[batch_indx] = active_len;

    uint32_t cnt = 0;
    for (uint32_t i = 0; i < label_len; i++) {
      if (cnt >= sparse_dim) {
        break;
      }
      active_neurons[batch_indx][cnt++] = labels[i];
      active_set.erase(labels[i]);
    }
    for (auto x : active_set) {
      if (cnt >= sparse_dim) {
        break;
      }
      active_neurons[batch_indx][cnt++] = x;
    }
  }
  std::fill_n(errors[batch_indx], dim, 0);
}

void SparseLayer::UpdateParameters(float lr, uint32_t iter, float B1, float B2,
                                   float eps) {
  float B1_ = 1 - pow(B1, iter);
  float B2_ = 1 - pow(B2, iter);

#pragma omp parallel for default(none) shared(lr, B1, B1_, B2, B2_, eps)
  for (uint64_t n = 0; n < dim; n++) {
    if (!is_active[n]) {
      continue;
    }

    for (uint64_t i = 0; i < prev_dim; i++) {
      auto indx = n * prev_dim + i;
      float grad = w_gradient[indx];
      w_momentum[indx] = B1 * w_momentum[indx] + (1 - B1) * grad;
      w_velocity[indx] = B2 * w_velocity[indx] + (1 - B2) * grad * grad;

      weights[indx] += lr * (w_momentum[indx] / B1_) /
                       (std::sqrt(w_velocity[indx] / B2_) + eps);

      w_gradient[indx] = 0;
    }

    float grad = b_gradient[n];
    b_momentum[n] = B1 * b_momentum[n] + (1 - B1) * grad;
    b_velocity[n] = B2 * b_velocity[n] + (1 - B2) * grad * grad;

    biases[n] +=
        lr * (b_momentum[n] / B1_) / (std::sqrt(b_velocity[n] / B2_) + eps);

    b_gradient[n] = 0;
    is_active[n] = false;
  }
}

void SparseLayer::BuildHashTables() {
  if (sparsity >= 1.0) {
    return;
  }
  uint64_t num_tables = hash_table->numTables();
  // TODO(nicholas): hashes could be array with size max(batch size, dim) that
  // is allocated once
  uint32_t* hashes = new uint32_t[num_tables * dim];

#pragma omp parallel for default(none) shared(num_tables, hashes)
  for (uint64_t n = 0; n < dim; n++) {
    hasher->hashSingleDense(weights + n * prev_dim, prev_dim,
                            hashes + n * num_tables);
  }

  hash_table->clearTables();
  hash_table->insertSequential(dim, 0, hashes);

  delete[] hashes;
}

void SparseLayer::ReBuildHashFunction() {
  if (sparsity >= 1.0) {
    return;
  }
  delete hasher;

  hasher = new utils::DWTAHashFunction(
      prev_dim, sampling_config.hashes_per_table, sampling_config.num_tables,
      sampling_config.range_pow);
}

void SparseLayer::SetBatchSize(uint64_t new_batch_size) {
  if (new_batch_size == batch_size) {
    return;
  }

  for (uint64_t batch = 0; batch < batch_size; batch++) {
    delete[] active_neurons[batch];
    delete[] activations[batch];
    delete[] errors[batch];
  }

  delete[] active_lens;
  delete[] active_neurons;
  delete[] activations;
  delete[] errors;

  batch_size = new_batch_size;

  active_lens = new uint32_t[batch_size];
  active_neurons = new uint32_t*[batch_size];
  activations = new float*[batch_size];
  errors = new float*[batch_size];

  for (uint64_t batch = 0; batch < batch_size; batch++) {
    active_neurons[batch] = new uint32_t[dim];
    activations[batch] = new float[dim];
    errors[batch] = new float[dim]();
  }
}

void SparseLayer::SetSparsity(float new_sparsity) {
  sparsity = new_sparsity;
  sparse_dim = sparsity * dim;
}

void SparseLayer::ShuffleRandNeurons() {
  if (sparsity < 1.0) {
    std::shuffle(rand_neurons, rand_neurons + dim, std::random_device{});
  }
}

float* SparseLayer::GetWeights() {
  float* weights_copy = new float[dim * prev_dim];
  std::copy(weights, weights + dim * prev_dim, weights_copy);

  return weights_copy;
}

float* SparseLayer::GetBiases() {
  float* biases_copy = new float[dim];
  std::copy(biases, biases + dim, biases_copy);

  return biases_copy;
}

SparseLayer::~SparseLayer() {
  for (uint64_t batch = 0; batch < batch_size; batch++) {
    delete[] active_neurons[batch];
    delete[] activations[batch];
    delete[] errors[batch];
  }

  delete[] active_lens;
  delete[] active_neurons;
  delete[] activations;
  delete[] errors;

  delete[] weights;
  delete[] w_gradient;
  delete[] w_momentum;
  delete[] w_velocity;

  delete[] biases;
  delete[] b_gradient;
  delete[] b_momentum;
  delete[] b_velocity;

  delete[] is_active;

  delete hasher;
  delete hash_table;
  delete[] rand_neurons;
}

}  // namespace thirdai::bolt
