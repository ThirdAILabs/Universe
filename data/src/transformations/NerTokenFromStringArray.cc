#include "NerTokenFromStringArray.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/columns/Column.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/Transformation.h>
#include <sstream>
#include <utility>

namespace thirdai::data {

NerTokenFromStringArray::NerTokenFromStringArray(
    std::string source_column, std::string token_column,
    std::string sentence_column, std::optional<std::string> target_column)
    : _source_column(std::move(source_column)),
      _token_column(std::move(token_column)),
      _sentence_column(std::move(sentence_column)),
      _target_column(std::move(target_column)) {}

std::vector<size_t> computeOffsets(
    const ArrayColumnBasePtr<std::string>& texts) {
  std::vector<size_t> offsets(texts->numRows() + 1);
  offsets[0] = 0;
  for (size_t i = 0; i < texts->numRows(); i++) {
    // number of samples is equal to number of tokens
    offsets[i + 1] = offsets[i] + texts->row(i).size();
  }
  return offsets;
}

ColumnMap NerTokenFromStringArray::apply(ColumnMap columns,
                                         State& state) const {
  (void)state;

  auto texts = columns.getArrayColumn<std::string>(_source_column);
  ValueColumnBasePtr<uint32_t> tags;
  if (_target_column) {
    tags = columns.getValueColumn<uint32_t>(*_target_column);
  }
  auto sample_offsets = computeOffsets(texts);

  std::vector<std::string> tokens(sample_offsets.back());

  std::vector<std::string> sentences(sample_offsets.back());

  std::vector<uint32_t> targets(sample_offsets.back());

  std::exception_ptr error;

#pragma omp parallel for default(none) \
    shared(texts, tags, tokens, sentences, targets, sample_offsets, error)
  for (size_t i = 0; i < texts->numRows(); i += 1) {
    try {
      auto row_tokens = texts->row(i);
      size_t sample_offset = sample_offsets[i];

      std::ostringstream oss;

      if (!(row_tokens.size() == 0)) {
        std::copy(row_tokens.begin(), row_tokens.end() - 1,
                  std::ostream_iterator<std::string>(oss, " "));
        oss << row_tokens[row_tokens.size() - 1];
      }

      std::string joinedString = oss.str();

      for (size_t start = 0; start < row_tokens.size(); start += 1) {
        sentences[sample_offset] = joinedString;
        tokens[sample_offset] = row_tokens[start];
        if (_target_column) {
          targets[sample_offset] = tags->row(i)[start];
        }
        sample_offset += 1;
      }

    } catch (...) {
#pragma omp critical
      error = std::current_exception();
    }
  }
  if (error) {
    std::rethrow_exception(error);
  }
  std::unordered_map<std::string, ColumnPtr> output_columns;
  output_columns[_token_column] =
      ValueColumn<std::string>::make(std::move(tokens));
  output_columns[_sentence_column] =
      ValueColumn<std::string>::make(std::move(sentences));
  if (_target_column) {
    output_columns[*_target_column] =
        ValueColumn<uint32_t>::make(std::move(targets), texts->dim());
  }

  return ColumnMap(output_columns);
}

ar::ConstArchivePtr NerTokenFromStringArray::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  map->set("source_column", ar::str(_source_column));
  map->set("token_column", ar::str(_token_column));
  map->set("sentence_column", ar::str(_sentence_column));
  if (_target_column) {
    map->set("target_column", ar::str(*_target_column));
  }

  return map;
}

NerTokenFromStringArray::NerTokenFromStringArray(const ar::Archive& archive)
    : _source_column(archive.str("source_column")),
      _token_column(archive.str("token_column")),
      _sentence_column(archive.str("sentence_column")),
      _target_column(archive.getOpt<ar::Str>("target_column")) {}
}  // namespace thirdai::data