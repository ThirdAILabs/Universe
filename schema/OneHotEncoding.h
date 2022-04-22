#pragma once

#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include "Schema.h"

namespace thirdai::schema {

struct OneHotEncodingBlock: public ABlock {
  OneHotEncodingBlock(const uint32_t col, const uint32_t out_dim, const uint32_t offset): _col(col), _out_dim(out_dim), _offset(offset) {}

  void consume(std::vector<std::string_view> line, InProgressVector &output_vec) override {
    auto num = getNumberU32(line[_col]);
    size_t hash = num % _out_dim + _offset;
    output_vec[hash]++;
  }

  static std::shared_ptr<ABlockBuilder> Builder(const uint32_t col, const uint32_t out_dim) {
    return std::make_shared<OneHotEncodingBlockBuilder>(col, out_dim);
  }
  struct OneHotEncodingBlockBuilder: public ABlockBuilder {
    OneHotEncodingBlockBuilder(const uint32_t col, const uint32_t out_dim)
    : _col(col), _out_dim(out_dim) {}

    std::unique_ptr<ABlock> build(uint32_t &offset) const override {
      auto built = std::make_unique<OneHotEncodingBlock>(_col, _out_dim, offset);
      offset += _out_dim;
      return built;
    }

    size_t maxColumn() const override { return _col; }

    size_t inputFeatDim() const override { return _out_dim; }

  private:
    const uint32_t _col;
    const uint32_t _out_dim;
  };

 private:


  const uint32_t _col;
  const uint32_t _out_dim;
  const uint32_t _offset;
  std::vector<std::pair<uint32_t, uint32_t>> _hash_constants;
};

} // namespace thirdai::schema
