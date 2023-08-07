#include "StringHash.h"
#include <hashing/src/MurmurHash.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ValueColumns.h>

namespace thirdai::data {

StringHash::StringHash(const proto::data::StringHash& string_hash)
    : _input_column_name(string_hash.input_column()),
      _output_column_name(string_hash.output_column()),
      _seed(string_hash.seed()) {
  if (string_hash.has_hash_range()) {
    _output_range = string_hash.hash_range();
  }
}

ColumnMap StringHash::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto column = columns.getValueColumn<std::string>(_input_column_name);

  std::vector<uint32_t> hashed_values(column->numRows());

#pragma omp parallel for default(none) shared(column, hashed_values)
  for (uint64_t i = 0; i < column->numRows(); i++) {
    hashed_values[i] = hash(column->value(i));
  }

  auto output_column =
      ValueColumn<uint32_t>::make(std::move(hashed_values), _output_range);

  columns.setColumn(_output_column_name, output_column);

  return columns;
}

uint32_t StringHash::hash(const std::string& str) const {
  uint32_t hash = hashing::MurmurHash(str.data(), str.length(), _seed);
  if (_output_range) {
    return hash % *_output_range;
  }
  return hash;
}

proto::data::Transformation* StringHash::toProto() const {
  auto* transformation = new proto::data::Transformation();
  auto* string_hash = transformation->mutable_string_hash();

  string_hash->set_input_column(_input_column_name);
  string_hash->set_output_column(_output_column_name);

  if (_output_range) {
    string_hash->set_hash_range(*_output_range);
  }

  string_hash->set_seed(_seed);

  return transformation;
}

}  // namespace thirdai::data