#include "NerTokenTagCounter.h"
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <data/src/transformations/Tabular.h>
#include <utils/text/StringManipulation.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace thirdai::data::ner {

TokenTagCounter::TokenTagCounter(
    uint32_t number_bins,
    std::unordered_map<std::string, uint32_t> tag_to_label)
    : _tag_to_label(std::move(tag_to_label)) {
  uint32_t max_label = 0;
  for (const auto& taglabel : _tag_to_label) {
    max_label = std::max(max_label, taglabel.second);
  }
  _num_unique_counters = max_label + 1;

  _tag_encoders = std::vector<NumericalColumn>(_num_unique_counters);
  for (const auto& taglabel : _tag_to_label) {
    _tag_encoders[taglabel.second] =
        NumericalColumn(taglabel.first, 0, 1, number_bins);
  }
}

void TokenTagCounter::addTokenTag(const std::string& token,
                                  const std::string& tag) {
  if (_tag_to_label.count(tag) == 0) {
    throw std::invalid_argument("The tag " + tag +
                                " is not present in the tag to label map.");
  }

  if (_token_counts.count(token)) {
    _token_counts[token][_tag_to_label[tag]]++;
    return;
  }

  _token_counts[token] = std::vector<uint32_t>(_num_unique_counters, 0);
  _token_counts[token][_tag_to_label[tag]]++;
}

int TokenTagCounter::getTokenTagCount(const std::string& token,
                                      const std::string& tag) const {
  if (_token_counts.count(token)) {
    return _token_counts.at(token).at(_tag_to_label.at(tag));
  }

  return 0;
}

float TokenTagCounter::getTokenTagRatio(const std::string& token,
                                        const std::string& tag) const {
  if (!_token_counts.count(token)) {
    return 0;
  }

  int count = getTokenTagCount(token, tag);

  uint32_t num_token_counter = 0;
  for (uint32_t i = 0; i < _num_unique_counters; i++) {
    num_token_counter += _token_counts.at(token)[i];
  }
  return num_token_counter > 0 ? static_cast<float>(count) / num_token_counter
                               : 0.0;
}

std::string TokenTagCounter::getTokenEncoding(const std::string& token) const {
  if (!_token_counts.count(token)) {
    // return " UNIQUE";
    return "";
  }

  std::string encoding;

  uint32_t num_token_counter = 0;
  for (uint32_t i = 0; i < _num_unique_counters; i++) {
    num_token_counter += _token_counts.at(token)[i];
  }

  for (uint32_t i = 0; i < _num_unique_counters; i++) {
    float ratio =
        num_token_counter > 0
            ? static_cast<float>(_token_counts.at(token)[i]) / num_token_counter
            : 0;
    encoding +=
        " " + std::to_string(_tag_encoders[i].encode(std::to_string(ratio)));
  }

  return encoding;
}

ar::ConstArchivePtr TokenTagCounter::toArchive() const {
  auto map = ar::Map::make();
  map->set("tag_to_label",
           ar::mapStrU64({_tag_to_label.begin(), _tag_to_label.end()}));
  map->set("num_unique_counters", ar::u64(_num_unique_counters));

  ar::MapStrVecU64 token_counts;
  for (const auto& [token, counts] : _token_counts) {
    for (const auto& count : counts) {
      token_counts[token].push_back(count);
    }
  }

  map->set("token_counts", ar::mapStrVecU64(std::move(token_counts)));

  auto numerical_columns = ar::List::make();
  for (const auto& num_col : _tag_encoders) {
    numerical_columns->append(num_col.toArchive());
  }
  map->set("tag_encoders", numerical_columns);

  return map;
}

TokenTagCounter::TokenTagCounter(const ar::Archive& archive) {
  const auto& tag_to_label = archive.getAs<ar::MapStrU64>("tag_to_label");
  _tag_to_label = {tag_to_label.begin(), tag_to_label.end()};

  _num_unique_counters = archive.u64("num_unique_counters");

  const auto& token_counts = archive.getAs<ar::MapStrVecU64>("token_counts");

  for (const auto& [token, counts] : token_counts) {
    std::vector<uint32_t> single_token_counts;
    single_token_counts.reserve(_num_unique_counters);
    for (const auto& count : counts) {
      single_token_counts.push_back(count);
    }
    _token_counts[token] = single_token_counts;
  }

  for (const auto& num_col : archive.get("tag_encoders")->list()) {
    _tag_encoders.push_back(NumericalColumn(*num_col));
  }
}

}  // namespace thirdai::data::ner