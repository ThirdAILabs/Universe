#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>

namespace thirdai::bolt::NER {

inline uint32_t getMaxLabelFromTagToLabel(
    std::unordered_map<std::string, uint32_t>& tag_to_label) {
  auto maxPair = std::max_element(
      tag_to_label.begin(), tag_to_label.end(),
      [](const auto& a, const auto& b) { return a.second < b.second; });
  return maxPair->second + 1;
}

}  // namespace thirdai::bolt::NER
