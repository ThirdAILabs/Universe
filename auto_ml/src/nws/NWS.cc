#include "NWS.h"
#include <cstdint>
#include <algorithm>
#include <iostream>

namespace thirdai::automl {

// For things like SRP that involves matrix multiplications, there are
// probably faster ways to parallelize this.
std::vector<uint32_t> SRP::hash(const std::vector<float>& input) const {
  assert(input.size() == _srp.inputDim());
  std::vector<uint32_t> output(rows());
  _srp.hashSingleDense(input.data(), input.size(), output.data());
  return output;
}

std::vector<uint32_t> L2::hash(const std::vector<float> &input) const {
  std::vector<uint32_t> hashes(_rows);
  for (uint32_t row = 0; row < _rows; row++) {
    std::vector<uint32_t> row_hashes(_hashes_per_row, 0);
    for (uint32_t i = 0; i < _hashes_per_row; i++) {
      const uint32_t idx = row * _hashes_per_row + i;
      // Implicit floor
      const int32_t signed_hash = (dot(_projections[idx], input) + _biases[idx]) / _scale;
      // Use bitwise OR rather than assignment to preserve bits
      row_hashes[i] |= signed_hash;
    }
    hashing::defaultCompactHashes(
      /* hashes= */ row_hashes.data(),
      /* output_hashes= */ &hashes[row],
      /* length_output= */ 1,
      /* hashes_per_output_value= */ row_hashes.size());
      hashes[row] %= _range;
  }
  return hashes;
}

std::vector<std::vector<float>> L2::make_projections(uint32_t input_dim, uint32_t hashes_per_row, uint32_t rows, uint32_t seed) {
  std::mt19937 rng;
  rng.seed(seed);
  std::normal_distribution<float> distribution(/* mean= */ 0.0, /* stddev= */ 1.0);

  std::vector<std::vector<float>> projections(hashes_per_row * rows);
  for (auto& projection : projections) {
    for (uint32_t i = 0; i < input_dim; i++) {
      projection.push_back(distribution(rng));
    }
  }

  return projections;
}

std::vector<float> L2::make_biases(uint32_t hashes_per_row, uint32_t rows, float scale, uint32_t seed) {
  std::mt19937 rng;
  rng.seed(seed * 3);  // Just so that it uses a different seed than the projections.
  std::uniform_real_distribution<float> distribution(/* min= */ 0, /* max= */ scale);

  std::vector<float> biases(hashes_per_row * rows);
  for (float& bias : biases) {
    bias = distribution(rng);
  }

  return biases;
}

float L2::dot(const std::vector<float>& a, const std::vector<float>& b) {
  assert(a.size() == b.size());
  float prod = 0;
  for (uint32_t i = 0; i < a.size(); i++) {
    prod += a[i] * b[i];
  }
  return prod;
}


void RACE::update(const std::vector<float>& key, const std::vector<float>& value) {
  assert(value.size() == _val_dim);
  uint32_t skip_buckets = 0;
  for (const uint32_t bucket : _hash->hash(key)) {
    const uint32_t start_idx = (skip_buckets + bucket) * _val_dim;
    for (uint32_t i = 0; i < _val_dim; i++) {
      _arrays[start_idx + i] += value[i];
    }
    skip_buckets += _hash->range();
  }
}

std::vector<float> RACE::query(const std::vector<float>& key) const {
  std::vector<float> value(_val_dim, 0.0);
  uint32_t skip_buckets = 0;
  for (const uint32_t bucket : _hash->hash(key)) {
    const uint32_t start_idx = (skip_buckets + bucket) * _val_dim;
    for (uint32_t i = 0; i < _val_dim; i++) {
      value[i] += _arrays[start_idx + i];
    }
    skip_buckets += _hash->range();
  }
  for (uint32_t i = 0; i < _val_dim; i++) {
    value[i] /= _hash->rows();
  }
  return value;
}

void RACE::debug(const std::vector<float>& key) const {
  uint32_t row = 0;
  uint32_t offset = 0;
  for (const uint32_t bucket : _hash->hash(key)) {
    std::cout << "row=" << row << " offset=" << offset << " bucket=" << bucket
              << " value=" << _arrays[offset + bucket] << std::endl;
    offset += _hash->range();
    row++;
  }
}

void RACE::merge(const RACE& other, uint32_t threads) {
  assert(other._hash == _hash);
  const uint32_t per_thread = (_arrays.size() + threads - 1) / threads;
#pragma omp parallel for default(none) shared(other, threads, per_thread)
  for (uint32_t thread = 0; thread < threads; thread++) {
    const uint32_t start = thread * per_thread;
    const uint32_t end = std::min<size_t>((thread + 1) * per_thread, _arrays.size());
    for (uint32_t i = start; i < end; i++) {
      _arrays[i] += other._arrays[i];
    }
  }
}

void RACE::print() const {
  uint32_t skip_buckets = 0;
  for (uint32_t row = 0; row < _hash->rows(); row++) {
    std::cout << "[ ";
    for (uint32_t bucket = 0; bucket < _hash->range(); bucket++) {
      std::cout << "[ ";
      for (uint32_t dim = 0; dim < _val_dim; dim++) {
        const uint32_t idx = skip_buckets * _val_dim + dim;
        std::cout << _arrays[idx] << " ";
      }
      std::cout << "] ";
      skip_buckets += 1;
    }
    std::cout << "]" << std::endl;
  }
}

void NadarayaWatsonSketch::train(const std::vector<std::vector<float>>& inputs,
                                 const std::vector<std::vector<float>>& outputs) {
  assert(inputs.size() == outputs.size());

  const std::vector<float> ones(_bottom.valDim(), 1.0);
  
  for (size_t i = 0; i < inputs.size(); i++) {
    _top.update(inputs[i], outputs[i]);
    _bottom.update(inputs[i], ones);
  }
}

void NadarayaWatsonSketch::trainParallel(
    const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<float>>& outputs, uint32_t threads) {
  assert(inputs.size() == outputs.size());

  const std::vector<float> ones(_bottom.valDim(), 1.0);

  std::vector<RACE> tops;
  std::vector<RACE> bottoms;
  tops.reserve(threads);
  bottoms.reserve(threads);
  for (uint32_t thread = 0; thread < threads; thread++) {
    tops.emplace_back(_top.hash(), _top.valDim());
    bottoms.emplace_back(_bottom.hash(), _bottom.valDim());
  }

  const uint32_t per_thread = (inputs.size() + threads - 1) / threads;

#pragma omp parallel for default(none) \
    shared(inputs, outputs, threads, tops, bottoms, per_thread, ones)
  for (uint32_t thread = 0; thread < threads; thread++) {
    const uint32_t start = thread * per_thread;
    const uint32_t end = std::min<size_t>((thread + 1) * per_thread, inputs.size());
    for (size_t i = start; i < end; i++) {
      tops[thread].update(inputs[i], outputs[i]);
      bottoms[thread].update(inputs[i], ones);
    }
  }

  for (const auto& top : tops) {
    _top.merge(top, threads);
  }
  for (const auto& bottom : bottoms) {
    _bottom.merge(bottom, threads);
  }
}

std::vector<std::vector<float>> NadarayaWatsonSketch::predict(
    const std::vector<std::vector<float>>& inputs) const {
  std::vector<std::vector<float>> outputs(inputs.size());
#pragma omp parallel for default(none) shared(inputs, outputs, std::cout)
  for (size_t i = 0; i < inputs.size(); i++) {
    outputs[i] = std::vector<float>(_top.valDim());
    auto top = _top.query(inputs[i]);
    auto bottom = _bottom.query(inputs[i]);
    for (uint32_t j = 0; j < _top.valDim(); j++) {
      outputs[i][j] = bottom[j] ? top[j] / bottom[j] : 0;
    }
  }
  return outputs;
}

std::vector<std::vector<float>> NadarayaWatsonSketch::predictDebug(
    const std::vector<std::vector<float>>& inputs) const {
  std::vector<std::vector<float>> outputs(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    outputs[i] = std::vector<float>(_top.valDim());
    auto top = _top.query(inputs[i]);
    auto bottom = _bottom.query(inputs[i]);
    for (uint32_t j = 0; j < _top.valDim(); j++) {
      outputs[i][j] = top[j] / bottom[j];
    }
    std::cout << "Top: ";
    for (uint32_t j = 0; j < _top.valDim(); j++) {
      std::cout << top[j] << " " << std::endl;
    } 
    std::cout << std::endl;
    std::cout << "Bottom: ";
    for (uint32_t j = 0; j < _bottom.valDim(); j++) {
      std::cout << bottom[j] << " " << std::endl;
    } 
    std::cout << std::endl;
  }
  return outputs;
}

}  // namespace thirdai::automl