#include "NWS.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>

namespace thirdai::automl {

MockHash::MockHash(
    std::vector<std::pair<std::vector<float>, std::vector<uint32_t>>>&&
        registered_hashes)
    : _inputs_and_hashes(std::move(registered_hashes)) {
  const size_t input_dim = _inputs_and_hashes.front().first.size();
  const size_t rows = _inputs_and_hashes.front().second.size();
  for (const auto& [input, hashes] : _inputs_and_hashes) {
    if (input.size() != input_dim) {
      throw std::invalid_argument("Inputs have inconsistent dimension.");
    }
    if (hashes.size() != rows) {
      throw std::invalid_argument(
          "Inputs have inconsistent numbers of hashes.");
    }
    for (const uint32_t hash : hashes) {
      _range = std::max(_range, hash + 1);
    }
  }
}

std::vector<uint32_t> MockHash::hash(const std::vector<float>& input) const {
  for (const auto& [registered_input, hashes] : _inputs_and_hashes) {
    if (input.size() != registered_input.size()) {
      throw std::invalid_argument("Input vector has wrong dimension.");
    }
    bool match = true;
    for (size_t i = 0; i < input.size(); i++) {
      if (input[i] != registered_input[i]) {
        match = false;
        continue;
      }
    }
    if (match) {
      return hashes;
    }
  }
  throw std::invalid_argument("Input vector is unregistered.");
}

// For things like SRP that involves matrix multiplications, there are
// probably faster ways to parallelize this.
std::vector<uint32_t> SRP::hash(const std::vector<float>& input) const {
  assert(input.size() == _srp.inputDim());
  std::vector<uint32_t> output(rows());
  _srp.hashSingleDense(input.data(), input.size(), output.data());
  return output;
}

std::vector<uint32_t> L2Hash::hash(const std::vector<float>& input) const {
  std::vector<uint32_t> hashes(_rows);
  for (size_t row = 0; row < _rows; row++) {
    std::vector<uint32_t> row_hashes(_hashes_per_row, 0);
    for (size_t i = 0; i < _hashes_per_row; i++) {
      const size_t idx = row * _hashes_per_row + i;
      // Need explicit floor so negative numbers are rounded correctly.
      // E.g. without the explicit floor, -1.5 is rounded to -1 instead of -2.
      const int32_t signed_hash =
          std::floor((dot(_projections[idx], input) + _biases[idx]) / _scale);
      // Use bitwise OR rather than assignment to preserve bits
      row_hashes[i] |= signed_hash;
    }
    hashing::defaultCompactHashes(
        /* hashes= */ row_hashes.data(),
        /* output_hashes= */ &hashes[row],
        /* length_output= */ 1,
        /* hashes_per_output_value= */ row_hashes.size());
    if (_range) {
      hashes[row] = (hashes[row] * _rehash_a[row] + _rehash_b[row]) %
                    _prime_mod % *_range;
    }
  }
  return hashes;
}

std::vector<std::vector<float>> L2Hash::make_projections(
    uint32_t input_dim, uint32_t hashes_per_row, uint32_t rows, uint32_t seed) {
  std::mt19937 rng;
  rng.seed(seed);
  std::normal_distribution<float> distribution(/* mean= */ 0.0,
                                               /* stddev= */ 1.0);

  std::vector<std::vector<float>> projections(hashes_per_row * rows);
  for (auto& projection : projections) {
    for (size_t i = 0; i < input_dim; i++) {
      projection.push_back(distribution(rng));
    }
  }

  return projections;
}

std::vector<float> L2Hash::make_biases(uint32_t hashes_per_row, uint32_t rows,
                                       float scale, uint32_t seed) {
  std::mt19937 rng;
  rng.seed(seed);
  std::uniform_real_distribution<float> distribution(/* min= */ 0,
                                                     /* max= */ scale);

  std::vector<float> biases(hashes_per_row * rows);
  for (float& bias : biases) {
    bias = distribution(rng);
  }

  return biases;
}

std::vector<uint32_t> L2Hash::make_random_ints(uint32_t size, uint32_t min,
                                               uint32_t seed) {
  std::mt19937 rng;
  rng.seed(
      seed);  // Just so that it uses a different seed than the projections.
  std::uniform_int_distribution<uint32_t> distribution(/* min= */ min,
                                                       /* max= */ _prime_mod);

  std::vector<uint32_t> ints(size);
  for (uint32_t& i : ints) {
    i = distribution(rng);
  }

  return ints;
}

float L2Hash::dot(const std::vector<float>& a, const std::vector<float>& b) {
  assert(a.size() == b.size());
  float prod = 0;
  for (size_t i = 0; i < a.size(); i++) {
    prod += a[i] * b[i];
  }
  return prod;
}

void RACE::update(const std::vector<float>& key, float value) {
  assert(value.size() == _val_dim);
  size_t skip_buckets = 0;
  for (const uint32_t bucket : _hash->hash(key)) {
    _arrays[skip_buckets + bucket] += value;
    skip_buckets += _hash->range();
  }
}

float RACE::query(const std::vector<float>& key) const {
  float value = 0;
  size_t skip_buckets = 0;
  for (const uint32_t bucket : _hash->hash(key)) {
    value += _arrays[skip_buckets + bucket];
    skip_buckets += _hash->range();
  }
  value /= _hash->rows();
  return value;
}

void RACE::debug(const std::vector<float>& key) const {
  size_t row = 0;
  size_t offset = 0;
  for (const uint32_t bucket : _hash->hash(key)) {
    std::cout << "row=" << row << " offset=" << offset << " bucket=" << bucket
              << " value=" << _arrays[offset + bucket] << std::endl;
    offset += _hash->range();
    row++;
  }
}

void RACE::merge(const RACE& other, uint32_t threads) {
  assert(other._hash == _hash);
  const size_t per_thread = (_arrays.size() + threads - 1) / threads;
#pragma omp parallel for default(none) shared(other, threads, per_thread)
  for (uint32_t thread = 0; thread < threads; thread++) {
    const size_t start = thread * per_thread;
    const size_t end =
        std::min<size_t>((thread + 1) * per_thread, _arrays.size());
    for (size_t i = start; i < end; i++) {
      _arrays[i] += other._arrays[i];
    }
  }
}

void RACE::print() const {
  for (size_t row = 0; row < _hash->rows(); row++) {
    const size_t skip_buckets = row * _hash->range();
    std::cout << "[ ";
    for (size_t bucket = 0; bucket < _hash->range(); bucket++) {
      std::cout << _arrays[skip_buckets + bucket] << " ";
    }
    std::cout << "]" << std::endl;
  }
}

void NadarayaWatsonSketch::train(const std::vector<std::vector<float>>& inputs,
                                 const std::vector<float>& outputs) {
  assert(inputs.size() == outputs.size());

  for (size_t i = 0; i < inputs.size(); i++) {
    _top.update(inputs[i], outputs[i]);
    _bottom.update(inputs[i], 1.0);
  }
}

void NadarayaWatsonSketch::trainParallel(
    const std::vector<std::vector<float>>& inputs,
    const std::vector<float>& outputs, uint32_t threads) {
  assert(inputs.size() == outputs.size());

  std::vector<RACE> tops;
  std::vector<RACE> bottoms;
  tops.reserve(threads);
  bottoms.reserve(threads);
  for (uint32_t thread = 0; thread < threads; thread++) {
    tops.emplace_back(_top.hash());
    bottoms.emplace_back(_bottom.hash());
  }

  const size_t per_thread = (inputs.size() + threads - 1) / threads;

#pragma omp parallel for default(none) \
    shared(inputs, outputs, threads, tops, bottoms, per_thread)
  for (uint32_t thread = 0; thread < threads; thread++) {
    const size_t start = thread * per_thread;
    const size_t end =
        std::min<size_t>((thread + 1) * per_thread, inputs.size());
    for (size_t i = start; i < end; i++) {
      tops[thread].update(inputs[i], outputs[i]);
      bottoms[thread].update(inputs[i], 1.0);
    }
  }

  for (const auto& top : tops) {
    _top.merge(top, threads);
  }
  for (const auto& bottom : bottoms) {
    _bottom.merge(bottom, threads);
  }
}

std::vector<float> NadarayaWatsonSketch::predict(
    const std::vector<std::vector<float>>& inputs) const {
  std::vector<float> outputs(inputs.size());
#pragma omp parallel for default(none) shared(inputs, outputs, std::cout)
  for (size_t i = 0; i < inputs.size(); i++) {
    auto top = _top.query(inputs[i]);
    auto bottom = _bottom.query(inputs[i]);
    outputs[i] = bottom ? top / bottom : 0;
  }
  return outputs;
}

std::vector<float> NadarayaWatsonSketch::predictDebug(
    const std::vector<std::vector<float>>& inputs) const {
  std::vector<float> outputs(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    auto top = _top.query(inputs[i]);
    auto bottom = _bottom.query(inputs[i]);
    outputs[i] = bottom ? top / bottom : 0;
    std::cout << "Top: " << top << std::endl;
    std::cout << "Bottom: " << bottom << std::endl;
  }
  return outputs;
}

}  // namespace thirdai::automl