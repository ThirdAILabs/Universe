#include "StringHash.h"
#include <hashing/src/MurmurHash.h>


namespace thirdai::dataset {

void StringHash::apply(ColumnMap& columns) {
  auto column = columns.getStringColumn(_input_column_name);

  std::vector<uint32_t> hashed_values(column->numRows());

#pragma omp parallel for default(none) \
    shared(column, hashed_values, column)
  for (uint64_t i = 0; i < column->numRows(); i++) {
    hashed_values[i] = hash((*column)[i]);
  }

  auto output_column = std::make_shared<VectorValueColumn<uint32_t>>(
      std::move(hashed_values), _output_range);

  columns.setColumn(_output_column_name, output_column);
}

uint32_t StringHash::hash(const std::string& str) const {
    return hashing::MurmurHash(str.data(), str.length(), _seed);
}

}  // namespace thirdai::dataset