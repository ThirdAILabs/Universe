#pragma once
#include <archive/src/Archive.h>
#include <data/src/transformations/Tabular.h>
#include <string>
#include <unordered_map>

namespace thirdai::data::ner {

class TokenTagCounter {
 public:
  explicit TokenTagCounter(
      uint32_t number_bins,
      std::unordered_map<std::string, uint32_t> tag_to_label);

  void addTokenTag(std::string token, const std::string& tag, bool lower_case);

  int getTokenTagCount(const std::string& token, const std::string& tag) const;

  float getTokenTagRatio(const std::string& token,
                         const std::string& tag) const;

  std::string getTokenEncoding(const std::string& token) const;

  explicit TokenTagCounter(const ar::Archive& archive);

  ar::ConstArchivePtr toArchive() const;

 private:
  std::unordered_map<std::string, uint32_t> _tag_to_label;
  uint32_t _num_unique_counters;
  std::unordered_map<std::string, std::vector<uint32_t>> _token_counts;
  std::vector<NumericalColumn> _tag_encoders;
};

using TokenTagCounterPtr = std::shared_ptr<TokenTagCounter>;
}  // namespace thirdai::data::ner