#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <string>
#include <unordered_map>

namespace thirdai::data {
inline std::vector<size_t> computeOffsets(
    const ArrayColumnBasePtr<std::string>& texts) {
  std::vector<size_t> offsets(texts->numRows() + 1);
  offsets[0] = 0;
  for (size_t i = 0; i < texts->numRows(); i++) {
    // number of samples is equal to number of tokens
    offsets[i + 1] = offsets[i] + texts->row(i).size();
  }
  return offsets;
}

class NerTokenFromStringArray final : public Transformation {
 public:
  NerTokenFromStringArray(
      std::string source_column, std::string token_column,
      std::string token_front_column, std::string token_behind_column,
      std::optional<std::string> target_column = std::nullopt,
      std::optional<std::unordered_map<std::string, uint32_t>> tag_to_label =
          std::nullopt);

  explicit NerTokenFromStringArray(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "ner_token_from_string_array"; }

 private:
  std::string _source_column;
  std::string _token_column;
  std::string _token_front_column;
  std::string _token_behind_column;
  std::optional<std::string> _target_column;
  std::optional<std::unordered_map<std::string, uint32_t>> _tag_to_label;
};
}  // namespace thirdai::data