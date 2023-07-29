#include "Sequence.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <hashing/src/MurmurHash.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/CsvParser.h>
#include <string>

namespace thirdai::data {

using dataset::parsers::CSV::parseLine;

Sequence::Sequence(std::string input_column_name,
                   std::string output_column_name, char delimiter, size_t dim)
    : _input_column_name(std::move(input_column_name)),
      _output_column_name(std::move(output_column_name)),
      _delimiter(delimiter),
      _dim(dim) {}

ColumnMap Sequence::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto sequences = columns.getValueColumn<std::string>(_input_column_name);

  std::vector<std::vector<uint32_t>> hashed_tokens(sequences->numRows());

#pragma omp parallel for default(none) shared(sequences, hashed_tokens)
  for (size_t i = 0; i < hashed_tokens.size(); i++) {
    auto elements = parseLine(sequences->value(i), _delimiter);

    std::vector<uint32_t> tokens;
    tokens.reserve(elements.size());
    for (size_t pos = 0; pos < elements.size(); pos++) {
      tokens.push_back(hash(elements[pos], pos));
    }

    hashed_tokens[i] = std::move(tokens);
  }

  auto output = ArrayColumn<uint32_t>::make(std::move(hashed_tokens), _dim);
  columns.setColumn(_output_column_name, output);

  return columns;
}

uint32_t Sequence::hash(const std::string& element, size_t position) const {
  return hashing::MurmurHash(element.data(), element.size(), position) % _dim;
}

template void Sequence::serialize(cereal::BinaryInputArchive&);
template void Sequence::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Sequence::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column_name,
          _output_column_name, _delimiter, _dim);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::Sequence)