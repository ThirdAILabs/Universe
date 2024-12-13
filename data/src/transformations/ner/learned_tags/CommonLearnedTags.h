#pragma once

#include "LearnedTag.h"
#include <memory>

namespace thirdai::data::ner {

inline NerTagPtr getLocationTag() {
  return NerLearnedTag::make("LOCATION",
                             /*supported_type=*/NerSupportedCharacterType::All,
                             /*consecutive_tags_required=*/3,
                             /*special_characters=*/{},
                             /*invalid_sizes=*/{});
}

inline NerTagPtr getSSNTag() {
  return NerLearnedTag::make(
      "SSN", /*supported_type=*/NerSupportedCharacterType::OnlyIntegers,
      /*consecutive_tags_required=*/1,
      /*special_characters=*/{}, /*invalid_sizes=*/{1, 6, 8},
      /*validation_pattern=*/
      R"(\b\d{3}([- .])\d{2}\1\d{4}|\b\d{3}\d{2}\d{4}\b)");
}

inline NerTagPtr getNameTag() {
  return NerLearnedTag::make(
      "NAME", /*supported_type=*/NerSupportedCharacterType::OnlyAlphabets,
      /*consecutive_tags_required=*/2,
      /*special_characters=*/{}, /*invalid_sizes=*/{});
}

inline NerTagPtr getLearnedTagFromString(const std::string& tag) {
  if (tag == "NAME") {
    return getNameTag();
  }
  if (tag == "SSN") {
    return getSSNTag();
  }
  if (tag == "LOCATION") {
    return getLocationTag();
  }

  return std::make_shared<NerLearnedTag>(tag);
}
}  // namespace thirdai::data::ner