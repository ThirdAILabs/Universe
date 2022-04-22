#pragma once

#include <cstdlib>
#include <memory>
#include "Schema.h"
#include <schema/InProgressVector.h>

namespace thirdai::schema {

struct NumberBlock: public ABlock {
  NumberBlock(const uint32_t col, const uint32_t offset): _col(col), _offset(offset) {}

  void extractFeatures(std::vector<std::string_view> line, InProgressSparseVector &vec) override {
    vec.addSingleFeature(_offset, getNumberU32(line[_col]));
  }
    
  static std::shared_ptr<ABlockConfig> Config(const uint32_t col) {
    return std::make_shared<NumberBlockConfig>(col);
  }
  struct NumberBlockConfig: public ABlockConfig {
    explicit NumberBlockConfig(const uint32_t col): _col(col) {}

    std::unique_ptr<ABlock> build(uint32_t &offset) const override {
      auto built = std::make_unique<NumberBlock>(_col, offset);
      offset++;
      return built;
    }

    size_t maxColumn() const override { return _col; }

    size_t featureDim() const override { return 1; }

  private:
    uint32_t _col;
  };

 private:

  uint32_t _col;
  uint32_t _offset;
};


} // namespace thirdai::schema