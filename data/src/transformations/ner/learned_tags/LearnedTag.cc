#include "LearnedTag.h"
#include <archive/src/Archive.h>
#include <data/src/transformations/ner/utils/utils.h>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
namespace thirdai::data::ner {

std::shared_ptr<NerTag> NerTag::fromArchive(const ar::Archive& archive) {
  auto type = static_cast<NerTagType>(archive.u64("type"));

  switch (type) {
    case NerTagType::NerLearnedTagType:
      return std::make_shared<NerLearnedTag>(archive);

    default:
      throw std::runtime_error("Unknown Tag Type " +
                               std::to_string(archive.u64("type")) + " found.");
  }
}

void NerLearnedTag::applyTypeFilter(
    utils::SentenceTags& sentence_tags,
    const std::vector<std::string>& tokens) const {
  if (_supported_type == NerSupportedCharacterType::All) {
    return;
  }

  if (_supported_type == NerSupportedCharacterType::OnlyIntegers) {
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

void NerLearnedTag::applyConsecutiveTagsAndValidationFilter(
    utils::SentenceTags& sentence_tags,
    const std::vector<std::string>& tokens) const {
  // the two filters are merged to avoid duplicate processing. both validation
  // and contiguous tag filter require finding the maximum window that have the
  // same tags.
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