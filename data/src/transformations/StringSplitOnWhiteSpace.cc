#include "StringSplitOnWhiteSpace.h"
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <utils/text/StringManipulation.h>
#include <string>
#include <utility>
#include <vector>

namespace thirdai::data {

// Note: This function is created separately instead of modifying the existing
// split function, since this function would almost
// double the memory usage, which can be a bottleneck for Inverted Index
std::pair<std::vector<std::string>, std::vector<std::pair<size_t, size_t>>>
splitOnWhiteSpaceWithOffsets(const std::string& text) {
  std::cerr << "LEN: " << text.size() << std::endl;

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
      offsets.emplace_back(word_start, i);
    }
    last_is_word = is_word;
  }
  if (last_is_word) {
    words.push_back(text.substr(word_start));
    offsets.emplace_back(word_start, text.size());
  }

  return {words, offsets};
}

std::pair<std::vector<std::string>, std::vector<std::pair<size_t, size_t>>>
splitOnWhiteSpaceWithOffsetsUnicode(const std::string& ascii_text) {
  auto text = text::toUnicode(ascii_text);

  std::vector<std::string> words;
  std::vector<std::pair<size_t, size_t>> offsets;

  bool last_is_word = false;
  size_t word_start = 0;

  for (size_t i = 0; i < text.size(); i++) {
    bool is_word = !text::isWhitespace(text[i]);
    if (!last_is_word && is_word) {
      word_start = i;
    } else if (last_is_word && !is_word) {
      words.push_back(
          text::fromUnicode(text.substr(word_start, i - word_start)));
      offsets.emplace_back(word_start, i);
    }
    last_is_word = is_word;
  }
  if (last_is_word) {
    words.push_back(text::fromUnicode(text.substr(word_start)));
    offsets.emplace_back(word_start, text.size());
  }

  return {words, offsets};
}

ColumnMap StringSplitOnWhiteSpace::apply(ColumnMap columns,
                                         State& state) const {
  (void)state;

  auto input_column = columns.getValueColumn<std::string>(_input_column_name);
  size_t num_rows = columns.numRows();

  std::vector<std::vector<std::string>> split_results(num_rows);
  std::vector<std::vector<std::pair<size_t, size_t>>> offset_results(num_rows);

#pragma omp parallel for default(none) \
    shared(split_results, offset_results, input_column, num_rows)
  for (size_t i = 0; i < num_rows; i++) {
    /**
     * Note: The unicode version returns offsets using the unicode characters,
     * this means that the offsets will not be correct if applied to the string
     * in ascii format.
     */
    auto [tokens, offsets] =
        _as_unicode
            ? splitOnWhiteSpaceWithOffsetsUnicode(input_column->value(i))
            : splitOnWhiteSpaceWithOffsets(input_column->value(i));

    split_results[i] = std::move(tokens);
    offset_results[i] = std::move(offsets);
  }

  auto output_column = ArrayColumn<std::string>::make(std::move(split_results));
  auto offset_column =
      ArrayColumn<std::pair<size_t, size_t>>::make(std::move(offset_results));

  columns.setColumn(_output_column_name, output_column);
  columns.setColumn(_output_column_name + "_offsets", offset_column);

  return columns;
}

ar::ConstArchivePtr StringSplitOnWhiteSpace::toArchive() const {
  auto map = ar::Map::make();
  map->set("type", ar::str(type()));
  map->set("input_column", ar::str(_input_column_name));
  map->set("output_column", ar::str(_output_column_name));
  map->set("as_unicode", ar::boolean(_as_unicode));
  return map;
}

StringSplitOnWhiteSpace::StringSplitOnWhiteSpace(const ar::Archive& archive)
    : _input_column_name(archive.str("input_column")),
      _output_column_name(archive.str("output_column")),
      _as_unicode(archive.getOr<ar::Boolean>("as_unicode", false)) {}

}  // namespace thirdai::data
