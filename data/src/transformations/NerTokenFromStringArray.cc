#include "NerTokenFromStringArray.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/Transformation.h>
#include <iterator>
#include <sstream>
#include <utility>

namespace thirdai::data {

NerTokenFromStringArray::NerTokenFromStringArray(
    std::string source_column, std::string token_column,
    std::string token_front_column, std::string token_behind_column,
    std::optional<std::string> target_column,
    std::optional<std::unordered_map<std::string, uint32_t>> tag_to_label)
    : _source_column(std::move(source_column)),
      _token_column(std::move(token_column)),
      _token_front_column(std::move(token_front_column)),
      _token_behind_column(std::move(token_behind_column)),
      _target_column(std::move(target_column)),
      _tag_to_label(std::move(tag_to_label)) {}

ColumnMap NerTokenFromStringArray::apply(ColumnMap columns,
                                         State& state) const {
  (void)state;

  auto texts = columns.getArrayColumn<std::string>(_source_column);
  ArrayColumnBasePtr<std::string> tags;
  if (_target_column) {
    tags = columns.getArrayColumn<std::string>(*_target_column);
  }
  auto sample_offsets = thirdai::data::computeOffsets(texts);

  std::vector<std::string> tokens(sample_offsets.back());

  std::vector<std::string> tokens_front(sample_offsets.back());
  std::vector<std::string> tokens_behind(sample_offsets.back());

  std::vector<uint32_t> targets(sample_offsets.back());

  std::exception_ptr error;

#pragma omp parallel for default(none)                                \
    shared(texts, tags, tokens, tokens_front, tokens_behind, targets, \
           sample_offsets, error)
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
        tokens_front[sample_offset] =
            (start + 1 < row_tokens.size()) ? row_tokens[start + 1] : "198";
        tokens_behind[sample_offset] =
            (start > 0) ? row_tokens[start - 1] : "198";
        tokens[sample_offset] = row_tokens[start];
        if (_target_column && _tag_to_label.has_value()) {
          const auto& tagLabelMap = _tag_to_label.value();
          const auto& tag = tags->row(i)[start];
          if (tagLabelMap.find(tag) != tagLabelMap.end()) {
            targets[sample_offset] = tagLabelMap.at(tag);
          } else {
            throw std::out_of_range("String not found in label map: " + tag);
          }
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
  output_columns[_token_front_column] =
      ValueColumn<std::string>::make(std::move(tokens_front));
  output_columns[_token_behind_column] =
      ValueColumn<std::string>::make(std::move(tokens_behind));
  if (_target_column && _tag_to_label.has_value()) {
    auto maxPair = std::max_element(
        _tag_to_label.value().begin(), _tag_to_label.value().end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    output_columns[*_target_column] =
        ValueColumn<uint32_t>::make(std::move(targets), maxPair->second + 1);
  }

  return ColumnMap(output_columns);
}

ar::ConstArchivePtr NerTokenFromStringArray::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  map->set("source_column", ar::str(_source_column));
  map->set("token_column", ar::str(_token_column));
  map->set("token_front_column", ar::str(_token_front_column));
  map->set("token_behind_column", ar::str(_token_behind_column));
  if (_target_column) {
    map->set("target_column", ar::str(*_target_column));
  }

  return map;
}

NerTokenFromStringArray::NerTokenFromStringArray(const ar::Archive& archive)
    : _source_column(archive.str("source_column")),
      _token_column(archive.str("token_column")),
      _token_front_column(archive.str("token_front_column")),
      _token_behind_column(archive.str("token_behind_column")),
      _target_column(archive.getOpt<ar::Str>("target_column")) {}
}  // namespace thirdai::data