#pragma once

#include "LearnedTag.h"
#include <memory>

namespace thirdai::data::ner {

inline NerTagPtr getLocationTag() {
  return NerLearnedTag::make("LOCATION", /*supported_types=*/2,
                             /*consecutive_tags_required=*/3,
                             /*special_characters=*/{},
                             /*invalid_sizes=*/{});
}

inline NerTagPtr getSSNTag() {
  return NerLearnedTag::make(
      "SSN", /*supported_types=*/0,
      /*consecutive_tags_required=*/1,
      /*special_characters=*/{}, /*invalid_sizes=*/{1, 6, 8},
      /*validation_pattern=*/
      R"(\b([0-9]{3})([- .]+)([0-9]{2})([- .]+)([0-9]{4})\b)");
}

inline NerTagPtr getNameTag() {
  return NerLearnedTag::make("NAME", /*supported_types=*/1,
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