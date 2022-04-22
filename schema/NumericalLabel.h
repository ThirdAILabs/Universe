#pragma once

#include <cstdlib>
#include <memory>
#include "Schema.h"

namespace thirdai::schema {

struct NumericalLabelBlock: public ABlock {
  explicit NumericalLabelBlock(uint32_t col): _col(col) {}

  void consume(std::vector<std::string_view> line, InProgressVector& output_vec) override {
    const char* start = line[_col].cbegin();
    char* end;
    uint32_t label = std::strtoul(start, &end, 10);
    output_vec.add_label(label);
  }

    static std::shared_ptr<ABlockBuilder> Builder(uint32_t col) {
      return std::make_shared<NumericalLabelBlockBuilder>(col);
    };
  struct NumericalLabelBlockBuilder: public ABlockBuilder {
    explicit NumericalLabelBlockBuilder(uint32_t col): _col(col) {}

    std::unique_ptr<ABlock> build(uint32_t &offset) const override {
      (void) offset;
      return std::make_unique<NumericalLabelBlock>(_col);
    }

    size_t maxColumn() const override { return _col; }

    size_t inputFeatDim() const override { return 0; }

  private:
    uint32_t _col;
  };

 private:
  
  uint32_t _col;
};


} // namespace thirdai::schema