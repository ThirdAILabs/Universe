#include "StringHash.h"
#include <hashing/src/MurmurHash.h>
#include <new_dataset/src/featurization_pipeline/Column.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
#include <memory>
#include <stdexcept>

namespace thirdai::data {

void StringHash::apply(ColumnMap& columns, bool prepare_for_backpropagate) {
  auto column = columns.getStringColumn(_input_column_name);

  std::vector<uint32_t> hashed_values(column->numRows());

#pragma omp parallel for default(none) shared(column, hashed_values)
  for (uint64_t i = 0; i < column->numRows(); i++) {
    hashed_values[i] = hash((*column)[i]);
  }

  if (prepare_for_backpropagate) {
    _hash_values = hashed_values;
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

void StringHash::backpropagate(ColumnMap& columns,
                               ContributionColumnMap& contribuition_columns) {
  auto column = columns.getStringColumn(_input_column_name);

  auto contribuition_column =
      contribuition_columns.getTokenContributionColumn(_output_column_name);

  if (!_hash_values) {
    throw std::invalid_argument(
        "in apply method didn't prepare for backpropagation.");
  }
  std::vector<std::vector<columns::Contribution<std::string>>> contributions(
      column->numRows());

#pragma omp parallel for default(none) \
    shared(contribuition_column, column, contributions)
  for (uint64_t i = 0; i < column->numRows(); i++) {
    std::vector<columns::Contribution<uint32_t>> row_token_contributions =
        contribuition_column->getRow(i);

    std::vector<columns::Contribution<std::string>> row_string_contributions = {
        columns::Contribution<std::string>(
            (*column)[i], row_token_contributions[0].gradient)};

    contributions[i] = row_string_contributions;
  }

  auto input_contribution_column =
      std::make_shared<columns::CppStringContributionColumn>(
          std::move(contributions));

  contribuition_columns.setColumn(_input_column_name,
                                  input_contribution_column);
}

}  // namespace thirdai::data