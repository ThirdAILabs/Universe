#include "LearnedTag.h"
#include <data/src/transformations/ner/utils/utils.h>
#include <regex>
#include <utility>
namespace thirdai::data::ner {

void NerLearnedTag::applyTypeFilter(
    utils::SentenceTags& sentence_tags,
    const std::vector<std::string>& tokens) const {
  if (_supported_types == ValidCharacterTypes::All) {
    return;
  }

  if (_supported_types == ValidCharacterTypes::OnlyIntegers) {
    for (size_t token_index = 0; token_index < sentence_tags.size();
         token_index++) {
      if (sentence_tags[token_index][0].first != _tag) {
        continue;
      }
      if (!utils::isNumberWithPunct(tokens[token_index], _special_characters) ||
          _invalid_sizes.count(tokens[token_index].size())) {
        sentence_tags[token_index][0] = std::make_pair("O", 1);
      }
    }

    return;
  }

  // if only alphabets are allowed
  for (size_t token_index = 0; token_index < sentence_tags.size();
       token_index++) {
    if (sentence_tags[token_index][0].first != _tag) {
      continue;
    }
    if (utils::containsNumbers(tokens[token_index], _special_characters) ||
        _invalid_sizes.count(tokens[token_index].size())) {
      sentence_tags[token_index][0] = std::make_pair("O", 1);
    }
  }
}

void NerLearnedTag::applyConsecutiveTagsFilter(
    utils::SentenceTags& sentence_tags,
    const std::vector<std::string>& tokens) const {
  size_t index = 0;
  while (index < sentence_tags.size()) {
    const std::string& tag =
        sentence_tags[index].empty() ? "" : sentence_tags[index][0].first;

    if (tag == _tag) {
      uint32_t maximum_consecutive_tags =
          _consecutive_tags_required > 1
              ? utils::find_max_contiguous_window(sentence_tags, index, tag) + 1
              : 1;

      std::string concatenated_tokens = tokens[index];
      for (uint32_t i = 1; i < maximum_consecutive_tags; ++i) {
        concatenated_tokens += " " + tokens[index + i];
      }

      if (maximum_consecutive_tags < _consecutive_tags_required ||
          (_validation_regex.has_value() and
           !std::regex_match(concatenated_tokens, _validation_regex.value()))) {
        for (uint32_t i = 0; i < maximum_consecutive_tags; ++i) {
          sentence_tags[index + i][0] = std::make_pair("O", 1);
        }
      }

      index += maximum_consecutive_tags;
    } else {
      ++index;
    }
  }
}

}  // namespace thirdai::data::ner