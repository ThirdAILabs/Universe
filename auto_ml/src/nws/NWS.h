#pragma once

#include <hashing/src/FastSRP.h>
#include <hashing/src/HashUtils.h>
#include <hashing/src/SRP.h>
#include <sys/types.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace thirdai::automl {

class Hash {
 public:
  virtual std::vector<uint32_t> hash(const std::vector<float>& input) const = 0;
  virtual std::string name() const = 0;
  virtual uint32_t rows() const = 0;
  virtual uint32_t range() const = 0;
  virtual ~Hash() {}
};

class SRP final : public Hash {
 public:
  SRP(uint32_t input_dim, uint32_t hashes_per_row, uint32_t rows, uint32_t seed)
      : _srp(input_dim, hashes_per_row, rows, seed) {}

  std::vector<uint32_t> hash(const std::vector<float>& input) const final;

  std::string name() const final { return "SRP"; }

  uint32_t rows() const final { return _srp.numTables(); }

  uint32_t range() const final { return _srp.range(); }

 private:
  hashing::SignedRandomProjection _srp;
};

class L2Hash final : public Hash {
 public:
  L2Hash(uint32_t input_dim, uint32_t hashes_per_row, uint32_t rows,
         float scale, uint32_t range, uint32_t seed)
      : _projections(make_projections(input_dim, hashes_per_row, rows, seed)),
        _biases(make_biases(hashes_per_row, rows, scale, seed)),
        _hashes_per_row(hashes_per_row),
        _rows(rows),
        _range(range),
        _scale(scale) {}

  std::vector<uint32_t> hash(const std::vector<float>& input) const final;

  std::string name() const final { return "L2"; }

  uint32_t rows() const final { return _rows; }

  uint32_t range() const final { return _range; }

 private:
  static std::vector<std::vector<float>> make_projections(
      uint32_t input_dim, uint32_t hashes_per_row, uint32_t rows,
      uint32_t seed);

  static std::vector<float> make_biases(uint32_t hashes_per_row, uint32_t rows,
                                        float scale, uint32_t seed);

  static float dot(const std::vector<float>& a, const std::vector<float>& b);

  std::vector<std::vector<float>> _projections;
  std::vector<float> _biases;
  uint32_t _hashes_per_row, _rows, _range;
  float _scale;
};

class RACE {
 public:
  explicit RACE(std::shared_ptr<Hash> hash, uint32_t val_dim)
      : _arrays(hash->rows() * hash->range() * val_dim),
        _hash(std::move(hash)),
        _val_dim(val_dim) {}

  void update(const std::vector<float>& key, const std::vector<float>& value);

  std::vector<float> query(const std::vector<float>& key) const;

  void debug(const std::vector<float>& key) const;

  void merge(const RACE& other, uint32_t threads);

  auto hash() { return _hash; }

  uint32_t valDim() const { return _val_dim; }

  void print() const;

 private:
  std::vector<float> _arrays;
  std::shared_ptr<Hash> _hash;
  uint32_t _val_dim;
};

class NadarayaWatsonSketch {
 public:
  explicit NadarayaWatsonSketch(const std::shared_ptr<Hash>& hash,
                                uint32_t val_dim)
      : _top(hash, val_dim), _bottom(hash, val_dim) {}

  void train(const std::vector<std::vector<float>>& inputs,
             const std::vector<std::vector<float>>& outputs);

  void trainParallel(const std::vector<std::vector<float>>& inputs,
                     const std::vector<std::vector<float>>& outputs,
                     uint32_t threads);

  std::vector<std::vector<float>> predict(
      const std::vector<std::vector<float>>& inputs) const;

  std::vector<std::vector<float>> predictDebug(
      const std::vector<std::vector<float>>& inputs) const;

 private:
  RACE _top, _bottom;
};

}  // namespace thirdai::automl