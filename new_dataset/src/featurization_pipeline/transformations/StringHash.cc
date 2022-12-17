#include "StringHash.h"
#include <hashing/src/MurmurHash.h>

namespace thirdai::data {

void StringHash::apply(ColumnMap& columns) {
  auto column = columns.getStringColumn(_input_column_name);

  std::vector<uint32_t> hashed_values(column->numRows());

#pragma omp parallel for default(none) shared(column, hashed_values)
  for (uint64_t i = 0; i < column->numRows(); i++) {
    hashed_values[i] = hash((*column)[i]);
  }

  auto output_column = std::make_shared<columns::CppTokenColumn>(
      std::move(hashed_values), _output_range);

  columns.setColumn(_output_column_name, output_column);
}

uint32_t StringHash::hash(const std::string& str) const {
  uint32_t hash = hashing::MurmurHash(str.data(), str.length(), _seed);
  if (_output_range) {
    return hash % *_output_range;
  }
  return hash;
}

void StringHash::backpropagate(
    ContributionColumnMap& /*contribuition_columns*/) {}

}  // namespace thirdai::data