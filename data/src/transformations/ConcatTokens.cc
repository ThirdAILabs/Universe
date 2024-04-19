#include "ConcatTokens.h"
#include <_types/_uint32_t.h>
#include <archive/src/Archive.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <optional>
#include <stdexcept>

namespace thirdai::data {

ColumnMap ConcatTokens::apply(ColumnMap columns, State& state) const {
  (void)state;

  size_t output_dim = 0;
  std::vector<ArrayColumnBasePtr<uint32_t>> input_cols;
  for (const auto& input_col : _input_cols) {
    auto col = columns.getArrayColumn<uint32_t>(input_col);
    if (!col->dim()) {
      throw std::invalid_argument(
          "Can only concatenate token columns with a dimension.");
    }
    output_dim += col->dim().value();
    input_cols.push_back(col);
  }

  std::vector<std::vector<uint32_t>> output(columns.numRows());

#pragma omp parallel for default(none) \
    shared(input_cols, output) if (output.size() > 1)
  for (size_t i = 0; i < output.size(); i++) {
    uint32_t offset = 0;
    for (const auto& col : input_cols) {
      for (uint32_t token : col->row(i)) {
        output[i].push_back(token + offset);
      }
      offset += col->dim().value();
    }
  }

  columns.setColumn(_output_col,
                    ArrayColumn<uint32_t>::make(std::move(output), output_dim));

  return columns;
}

ar::ConstArchivePtr ConcatTokens::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  map->set("input_cols", ar::vecStr(_input_cols));
  map->set("output_col", ar::str(_output_col));

  return map;
}

ConcatTokens::ConcatTokens(const ar::Archive& archive)
    : ConcatTokens(archive.getAs<ar::VecStr>("input_cols"),
                   archive.str("output_col")) {}

}  // namespace thirdai::data