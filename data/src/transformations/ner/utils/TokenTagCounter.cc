#include "TokenTagCounter.h"
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

  _token_counts[token]++;
  _total_tokens++;

  if (_token_tag_counts.count(token)) {
    _token_tag_counts[token][_tag_to_label[tag]]++;
    return;
  }

  _token_tag_counts[token] = std::vector<uint32_t>(_num_unique_counters, 0);
  _token_tag_counts[token][_tag_to_label[tag]]++;
}

std::string TokenTagCounter::getTokenEncoding(const std::string& token) const {
  auto it = _token_counts.find(token);
  if (it == _token_counts.end() ||
      (static_cast<double>(it->second) / _total_tokens) < DEFAULT_RARE_RATIO) {
    return " RARE_TOKEN";
  }

  std::string encoding;
  uint32_t total_token_count = _token_counts.at(token);
  for (uint32_t i = 0; i < _num_unique_counters; i++) {
    float ratio =
        static_cast<float>(_token_tag_counts.at(token)[i]) / total_token_count;
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

  ar::MapStrVecU64 token_tag_counts;
  for (const auto& [token, counts] : _token_tag_counts) {
    for (const auto& count : counts) {
      token_tag_counts[token].push_back(count);
    }
  }

  map->set("token_tag_counts", ar::mapStrVecU64(std::move(token_tag_counts)));

  auto numerical_columns = ar::List::make();
  for (const auto& num_col : _tag_encoders) {
    numerical_columns->append(num_col.toArchive());
  }
  map->set("tag_encoders", numerical_columns);

  map->set("token_counts", ar::mapStrU32(_token_counts));
  map->set("total_tokens", ar::u64(_total_tokens));
  return map;
}

TokenTagCounter::TokenTagCounter(const ar::Archive& archive) {
  const auto& tag_to_label = archive.getAs<ar::MapStrU64>("tag_to_label");
  _tag_to_label = {tag_to_label.begin(), tag_to_label.end()};

  _num_unique_counters = archive.u64("num_unique_counters");

  const auto& token_tag_counts =
      archive.getAs<ar::MapStrVecU64>("token_tag_counts");

  for (const auto& [token, counts] : token_tag_counts) {
    std::vector<uint32_t> single_token_tag_counts;
    single_token_tag_counts.reserve(_num_unique_counters);
    for (const auto& count : counts) {
      single_token_tag_counts.push_back(count);
    }
    _token_tag_counts[token] = single_token_tag_counts;
  }

  for (const auto& num_col : archive.get("tag_encoders")->list()) {
    _tag_encoders.push_back(NumericalColumn(*num_col));
  }

  _token_counts = archive.getAs<ar::MapStrU32>("token_counts");
  _total_tokens = archive.u64("total_tokens");
}

}  // namespace thirdai::data::ner