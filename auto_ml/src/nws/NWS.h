#pragma once

#include <hashing/src/FastSRP.h>
#include <hashing/src/HashUtils.h>
#include <hashing/src/SRP.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thirdai::automl {

class Hash {
 public:
  virtual std::vector<uint32_t> hash(const std::vector<float>& input) const = 0;
  virtual uint32_t hashAt(const std::vector<float>& input, uint32_t row) const = 0;
  virtual std::string name() const = 0;
  virtual size_t rows() const = 0;
  virtual size_t range() const = 0;
  virtual ~Hash() {}
};

class MockHash final : public Hash {
 public:
  explicit MockHash(
      std::vector<std::pair<std::vector<float>, std::vector<uint32_t>>>&&
          registered_hashes);

  std::vector<uint32_t> hash(const std::vector<float>& input) const final;

  uint32_t hashAt(const std::vector<float>& input, uint32_t row) const final;

  std::string name() const final { return "MockHash"; }

  size_t rows() const final { return _inputs_and_hashes.front().second.size(); }

  size_t range() const final { return _range; }

 private:
  std::vector<std::pair<std::vector<float>, std::vector<uint32_t>>>
      _inputs_and_hashes;
  uint32_t _range = 0;
};

class SRP final : public Hash {
 public:
  SRP(uint32_t input_dim, uint32_t hashes_per_row, uint32_t rows, uint32_t seed)
      : _srp(input_dim, hashes_per_row, rows, seed) {}

  std::vector<uint32_t> hash(const std::vector<float>& input) const final;

  uint32_t hashAt(const std::vector<float>& input, uint32_t row) const final;

  std::string name() const final { return "SRP"; }

  size_t rows() const final { return _srp.numTables(); }

  size_t range() const final { return _srp.range(); }

 private:
  hashing::SignedRandomProjection _srp;
};

class L2Hash final : public Hash {
 public:
  L2Hash(uint32_t input_dim, uint32_t hashes_per_row, uint32_t rows,
         float scale, uint32_t seed, std::optional<uint32_t> range)
      : _projections(make_projections(input_dim, hashes_per_row, rows, seed)),
        _biases(make_biases(hashes_per_row, rows, scale, seed * 3)),
        _hashes_per_row(hashes_per_row),
        _rows(rows),
        _rehash_a(make_random_ints(/* size= */ rows, /* min= */ 1, seed * 9)),
        _rehash_b(make_random_ints(/* size= */ rows, /* min= */ 0, seed * 27)),
        _scale(scale),
        _range(range) {}

  std::vector<uint32_t> hash(const std::vector<float>& input) const final;

  uint32_t hashAt(const std::vector<float>& input, uint32_t row) const final;

  std::string name() const final { return "L2"; }

  size_t rows() const final { return _rows; }

  size_t range() const final {
    return _range.value_or(static_cast<size_t>(1) << 32);
  }

 private:
  static std::vector<std::vector<float>> make_projections(
      uint32_t input_dim, uint32_t hashes_per_row, uint32_t rows,
      uint32_t seed);

  static std::vector<float> make_biases(uint32_t hashes_per_row, uint32_t rows,
                                        float scale, uint32_t seed);

  static std::vector<uint32_t> make_random_ints(uint32_t size, uint32_t min,
                                                uint32_t seed);

  static float dot(const std::vector<float>& a, const std::vector<float>& b);

  std::vector<std::vector<float>> _projections;
  std::vector<float> _biases;
  uint32_t _hashes_per_row, _rows;
  static constexpr uint32_t _prime_mod = 2147483647;  // 2^31 - 1
  std::vector<uint32_t> _rehash_a;
  std::vector<uint32_t> _rehash_b;
  float _scale;
  std::optional<uint32_t> _range;
};

class RACE {
 public:
  explicit RACE(std::shared_ptr<Hash> hash, bool sparse=false)
      : _arrays(sparse ? 0 : hash->rows() * hash->range()),
        _sparse_arrays(sparse ? hash->rows() : 0),
        _hash(std::move(hash)) {}

  void update(const std::vector<std::vector<float>>& keys, const std::vector<float>& values);

  float query(const std::vector<float>& key) const;

  void debug(const std::vector<float>& key) const;

  auto hash() { return _hash; }

  void print() const;

  size_t bytesUsed() const {
    if (!_arrays.empty()) {
      return _arrays.size() * 4;
    }
    size_t bytes = 0;
    for (const auto& map : _sparse_arrays) {
      // 4 bytes for key, 4 bytes for value.
      // Doesn't account for other data structure overheads.
      bytes += map.size() * 8;
    }
    return bytes;
  }

 private:
  std::vector<float> _arrays;
  std::vector<std::unordered_map<uint32_t, float>> _sparse_arrays;
  std::shared_ptr<Hash> _hash;
};

class NadarayaWatsonSketch {
 public:
  explicit NadarayaWatsonSketch(const std::shared_ptr<Hash>& hash, bool sparse)
      : _top(hash, sparse), _bottom(hash, sparse) {}

  void train(const std::vector<std::vector<float>>& inputs,
             const std::vector<float>& outputs);

  std::vector<float> predict(
      const std::vector<std::vector<float>>& inputs) const;

  std::vector<float> predictDebug(
      const std::vector<std::vector<float>>& inputs) const;
  
  size_t bytesUsed() const { return _top.bytesUsed() + _bottom.bytesUsed(); }

 private:
  RACE _top, _bottom;
};

}  // namespace thirdai::automl