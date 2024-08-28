#pragma once

#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <string>
#include <vector>

namespace thirdai::data {

class StringSplitOnWhiteSpace final : public Transformation {
 public:
  StringSplitOnWhiteSpace(std::string input_column_name,
                          std::string output_column_name)
      : _input_column_name(std::move(input_column_name)),
        _output_column_name(std::move(output_column_name)) {}

  explicit StringSplitOnWhiteSpace(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "string_split_whitespace"; }

 private:
  static std::pair<std::vector<std::string>,
                   std::vector<std::pair<size_t, size_t>>>
  splitOnWhiteSpaceWithOffsets(const std::string& text) {
    std::vector<std::string> words;
    std::vector<std::pair<size_t, size_t>> offsets;

    bool last_is_word = false;
    size_t word_start = 0;
    for (size_t i = 0; i < text.size(); i++) {
      bool is_word = !std::isspace(text[i]);
      if (!last_is_word && is_word) {
        word_start = i;
      } else if (last_is_word && !is_word) {
        words.push_back(text.substr(word_start, i - word_start));
        offsets.emplace_back(word_start, i - 1);
      }
      last_is_word = is_word;
    }
    if (last_is_word) {
      words.push_back(text.substr(word_start));
      offsets.emplace_back(word_start, text.size() - 1);
    }

    return {words, offsets};
  }

  std::string _input_column_name;
  std::string _output_column_name;
};

}  // namespace thirdai::data
