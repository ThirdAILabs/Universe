#pragma once

#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include "Schema.h"

namespace thirdai::schema {

struct FeatureHashingBlock: public ABlock {
  FeatureHashingBlock(const uint32_t col, const uint32_t n_hashes, const uint32_t out_dim, const uint32_t offset): _col(col), _n_hashes(n_hashes), _out_dim(out_dim), _offset(offset), _hash_constants(n_hashes) {
    std::srand(314152);
    for (size_t i = 0; i < _n_hashes; ++i) {
      _hash_constants[i] = std::make_pair(std::rand(), std::rand());
    }
  }

  void consume(std::vector<std::string_view> line, InProgressVector &output_vec) override {
    auto num = getNumberU32(line[_col]);
    for (size_t i = 0; i < _n_hashes; ++i) {
      size_t hash = (_hash_constants[i].first * num + _hash_constants[i].second) % _out_dim + _offset;
      output_vec[hash]++;
    }
  }

  static std::shared_ptr<ABlockBuilder> Builder(const uint32_t col, const uint32_t n_hashes, const uint32_t out_dim) {
    return std::make_shared<FeatureHashingBlockBuilder>(col, n_hashes, out_dim);
  }
  struct FeatureHashingBlockBuilder: public ABlockBuilder {
    FeatureHashingBlockBuilder(const uint32_t col, const uint32_t n_hashes, const uint32_t out_dim)
    : _col(col), _n_hashes(n_hashes), _out_dim(out_dim) {}

    std::unique_ptr<ABlock> build(uint32_t &offset) const override {
      auto built = std::make_unique<FeatureHashingBlock>(_col, _n_hashes, _out_dim, offset);
      offset += _out_dim;
      return built;
    }

    size_t maxColumn() const override { return _col; }

    size_t inputFeatDim() const override { return _out_dim; }

  private:
    const uint32_t _col;
    const uint32_t _n_hashes;
    const uint32_t _out_dim;
  };

 private:


  const uint32_t _col;
  const uint32_t _n_hashes;
  const uint32_t _out_dim;
  const uint32_t _offset;
  std::vector<std::pair<uint32_t, uint32_t>> _hash_constants;
};

} // namespace thirdai::schema
