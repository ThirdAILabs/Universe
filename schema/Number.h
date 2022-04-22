#pragma once

#include <cstdlib>
#include <memory>
#include "Schema.h"

namespace thirdai::schema {

struct NumberBlock: public ABlock {
  NumberBlock(const uint32_t col, const uint32_t offset): _col(col), _offset(offset) {}

  void consume(std::vector<std::string_view> line, InProgressVector &output_vec) override {
    output_vec[_offset] = getNumberU32(line[_col]);
  }
    
  static std::shared_ptr<ABlockBuilder> Builder(const uint32_t col) {
    return std::make_shared<NumberBlockBuilder>(col);
  }
  struct NumberBlockBuilder: public ABlockBuilder {
    explicit NumberBlockBuilder(const uint32_t col): _col(col) {}

    std::unique_ptr<ABlock> build(uint32_t &offset) const override {
      auto built = std::make_unique<NumberBlock>(_col, offset);
      offset++;
      return built;
    }

    size_t maxColumn() const override { return _col; }

    size_t inputFeatDim() const override { return 1; }

  private:
    uint32_t _col;
  };

 private:

  uint32_t _col;
  uint32_t _offset;
};


} // namespace thirdai::schema