#include "TokenLabelCounter.h"
#include "utils.h"
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <data/src/transformations/Tabular.h>
#include <utils/text/StringManipulation.h>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace thirdai::data::ner {

constexpr float DEFAULT_RARE_RATIO = 1e-6;

TokenLabelCounter::TokenLabelCounter(uint32_t number_bins, uint32_t num_labels)
    : _number_bins(number_bins), _num_labels(num_labels) {}

void TokenLabelCounter::addTokenLabel(const std::string& token,
                                      uint32_t label) {
  if (utils::isNumberWithPunct(token, {})) {
    return;
  }
  if (label >= _num_labels) {
    std::stringstream error;
    error << "Cannot add frequency count for the label " << label
          << " for a token label counter with " << _num_labels << " labels.";
    throw std::invalid_argument(error.str());
  }

  _token_counts[token]++;
  _total_tokens++;

  if (!_token_label_counts.count(token)) {
    _token_label_counts[token] = std::vector<uint32_t>(_num_labels, 0);
  }
  _token_label_counts[token][label]++;
}

std::string TokenLabelCounter::getTokenEncoding(
    const std::string& token) const {
  auto it = _token_counts.find(token);
  if (it == _token_counts.end() ||
      (static_cast<double>(it->second) / _total_tokens) < DEFAULT_RARE_RATIO) {
    return " RARE_TOKEN";
  }

  std::string encoding;
  uint32_t total_token_count = _token_counts.at(token);
  for (uint32_t i = 0; i < _num_labels; i++) {
    float ratio = static_cast<float>(_token_label_counts.at(token)[i]) /
                  total_token_count;
    uint32_t bin_id = ratio * _number_bins;
    encoding += " ratio_label_" + std::to_string(i * (_number_bins) + bin_id);
  }

  return encoding;
}

ar::ConstArchivePtr TokenLabelCounter::toArchive() const {
  auto map = ar::Map::make();
  map->set("number_bins", ar::u64(_number_bins));
  map->set("num_labels", ar::u64(_num_labels));

  ar::MapStrVecU64 token_label_counts;
  for (const auto& [token, counts] : _token_label_counts) {
    for (const auto& count : counts) {
      token_label_counts[token].push_back(count);
    }
  }

  map->set("token_label_counts",
           ar::mapStrVecU64(std::move(token_label_counts)));

  map->set("token_counts", ar::mapStrU32(_token_counts));
  map->set("total_tokens", ar::u64(_total_tokens));
  return map;
}

TokenLabelCounter::TokenLabelCounter(const ar::Archive& archive) {
  _number_bins = archive.u64("number_bins");
  _num_labels = archive.u64("num_labels");

  const auto& token_label_counts =
      archive.getAs<ar::MapStrVecU64>("token_label_counts");

  for (const auto& [token, counts] : token_label_counts) {
    std::vector<uint32_t> single_token_label_counts;
    for (const auto& count : counts) {
      single_token_label_counts.push_back(count);
    }
    _token_label_counts[token] = single_token_label_counts;
  }

  _token_counts = archive.getAs<ar::MapStrU32>("token_counts");
  _total_tokens = archive.u64("total_tokens");
}

}  // namespace thirdai::data::ner