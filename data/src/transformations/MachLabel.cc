#include "MachLabel.h"
#include <_types/_uint32_t.h>
#include <data/src/columns/ArrayColumns.h>

namespace thirdai::data {

MachLabel::MachLabel(std::string input_column, std::string output_column,
                     MachIndexPtr index)
    : _index(std::move(index)),
      _input_column(std::move(input_column)),
      _output_column(std::move(output_column)) {}

void MachLabel::setIndex(const MachIndexPtr& index) {
  if (_index->numBuckets() != index->numBuckets()) {
    throw std::invalid_argument(
        "Output range mismatch in new index. Index output range should be " +
        std::to_string(_index->numBuckets()) +
        " but provided an index with range = " +
        std::to_string(index->numBuckets()) + ".");
  }

  _index = index;
}

ColumnMap MachLabel::apply(ColumnMap columns) const {
  auto entities_column = columns.getArrayColumn<uint32_t>(_input_column);

  std::vector<std::vector<uint32_t>> hashes(entities_column->numRows());

#pragma omp parallel for default(none) shared(entities_column, hashes)
  for (size_t i = 0; i < entities_column->numRows(); i++) {
    for (uint32_t entity : entities_column->row(i)) {
      auto entity_hashes = _index->getHashes(entity);

      hashes[i].insert(hashes[i].end(), entity_hashes.begin(),
                       entity_hashes.end());
    }
  }

  auto output_column =
      ArrayColumn<uint32_t>::make(std::move(hashes), _index->numBuckets());
  columns.setColumn(_output_column, output_column);

  return columns;
}

}  // namespace thirdai::data