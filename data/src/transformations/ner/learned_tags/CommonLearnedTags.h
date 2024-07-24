#pragma once

#include "LearnedTag.h"

namespace thirdai::data::ner {

inline NerLearnedTag getLocationTag() {
  return NerLearnedTag("LOCATION", /*supported_types=*/2,
                       /*consecutive_tags_required=*/3,
                       /*special_characters=*/{}, /*invalid_sizes=*/{});
}

inline NerLearnedTag getSSNTag() {
  return NerLearnedTag("SSN", /*supported_types=*/0,
                       /*consecutive_tags_required=*/1,
                       /*special_characters=*/{}, /*invalid_sizes=*/{1, 6, 8},
                       /*validation_pattern=*/
                       R"(\b([0-9]{3})([- .]+)([0-9]{2})([- .]+)([0-9]{4})\b)");
}

inline NerLearnedTag getNameTag() {
  return NerLearnedTag("NAME", /*supported_types=*/1,
                       /*consecutive_tags_required=*/2,
                       /*special_characters=*/{}, /*invalid_sizes=*/{});
}

inline NerLearnedTag getLearnedTagFromString(const std::string& tag) {
  if (tag == "NAME") {
    return getNameTag();
  }
  if (tag == "SSN") {
    return getSSNTag();
  }
  if (tag == "LOCATION") {
    return getLocationTag();
  }

  return NerLearnedTag(tag);
}
}  // namespace thirdai::data::ner