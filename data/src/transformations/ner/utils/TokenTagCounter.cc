#include "TokenTagCounter.h"
#include "utils.h"
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <data/src/transformations/Tabular.h>
#include <utils/text/StringManipulation.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace thirdai::data::ner {

constexpr float DEFAULT_RARE_RATIO = 1e-6;

TokenTagCounter::TokenTagCounter(
    uint32_t number_bins,
    utils::TagTrackerPtr tag_tracker)
    : _number_bins(number_bins),
      _tag_tracker(std::move(tag_tracker)){
  
}

void TokenTagCounter::addTokenTag(const std::string& token,
                                  const std::string& tag) {
  if (utils::isNumberWithPunct(token, {})) {
    return;
  }
  if (!_tag_tracker->tagExists(tag)) {
    throw std::invalid_argument("The tag " + tag +
                                " is not present in the tag to label map.");
  }

  _token_counts[token]++;
  _total_tokens++;

  if (!_token_tag_counts.count(token)) {
    _token_tag_counts[token] = std::vector<uint32_t>(_tag_tracker->numLabels(), 0);
  }

  _token_tag_counts[token][_tag_tracker->tag_to_label(tag)]++;
}

std::string TokenTagCounter::getTokenEncoding(const std::string& token) const {
  auto it = _token_counts.find(token);
  if (it == _token_counts.end() ||
      (static_cast<double>(it->second) / _total_tokens) < DEFAULT_RARE_RATIO) {
    return " RARE_TOKEN";
  }

  std::string encoding;
  uint32_t total_token_count = _token_counts.at(token);
  for (uint32_t i = 0; i < _tag_tracker->numLabels(); i++) {
    float ratio =
        static_cast<float>(_token_tag_counts.at(token)[i]) / total_token_count;
    uint32_t bin_id = ratio * _number_bins;
    encoding += " ratio_label_" + std::to_string(i * (_number_bins) + bin_id);
  }

  return encoding;
}

ar::ConstArchivePtr TokenTagCounter::toArchive() const {
  auto map = ar::Map::make();
  map->set("number_bins", ar::u64(_number_bins));

  ar::MapStrVecU64 token_tag_counts;
  for (const auto& [token, counts] : _token_tag_counts) {
    for (const auto& count : counts) {
      token_tag_counts[token].push_back(count);
    }
  }

  map->set("token_tag_counts", ar::mapStrVecU64(std::move(token_tag_counts)));

  map->set("token_counts", ar::mapStrU32(_token_counts));
  map->set("total_tokens", ar::u64(_total_tokens));
  return map;
}

TokenTagCounter::TokenTagCounter(const ar::Archive& archive) {
  _number_bins = archive.u64("number_bins");

  const auto& token_tag_counts =
      archive.getAs<ar::MapStrVecU64>("token_tag_counts");

  for (const auto& [token, counts] : token_tag_counts) {
    std::vector<uint32_t> single_token_tag_counts;
    for (const auto& count : counts) {
      single_token_tag_counts.push_back(count);
    }
    _token_tag_counts[token] = single_token_tag_counts;
  }

  _token_counts = archive.getAs<ar::MapStrU32>("token_counts");
  _total_tokens = archive.u64("total_tokens");
}

}  // namespace thirdai::data::ner