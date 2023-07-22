#include "MachLabel.h"
#include <_types/_uint32_t.h>
#include <data/src/columns/ArrayColumns.h>

namespace thirdai::data {

MachLabel::MachLabel(std::string input_column, std::string output_column)
    : _input_column(std::move(input_column)),
      _output_column(std::move(output_column)) {}

ColumnMap MachLabel::apply(ColumnMap columns, State& state) const {
  auto entities_column = columns.getArrayColumn<uint32_t>(_input_column);

  std::vector<std::vector<uint32_t>> hashes(entities_column->numRows());

  const auto& index = state.machIndex();

#pragma omp parallel for default(none) shared(entities_column, hashes, index)
  for (size_t i = 0; i < entities_column->numRows(); i++) {
    for (uint32_t entity : entities_column->row(i)) {
      auto entity_hashes = index->getHashes(entity);

      hashes[i].insert(hashes[i].end(), entity_hashes.begin(),
                       entity_hashes.end());
    }
  }

  auto output_column =
      ArrayColumn<uint32_t>::make(std::move(hashes), index->numBuckets());
  columns.setColumn(_output_column, output_column);

  return columns;
}

}  // namespace thirdai::data