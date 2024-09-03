#pragma once
#include <archive/src/Archive.h>
#include <data/src/transformations/Tabular.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace thirdai::data::ner {
class TokenTagCounter {
 public:
  TokenTagCounter(uint32_t number_bins,
                  std::unordered_map<std::string, uint32_t> tag_to_label);

  void addTokenTag(const std::string& token, const std::string& tag);

  std::string getTokenEncoding(const std::string& token) const;

  explicit TokenTagCounter(const ar::Archive& archive);

  ar::ConstArchivePtr toArchive() const;

  void addTagLabel(const std::string& tag, uint32_t label) {
    _tag_to_label[tag] = label;

    bool requires_new_counter = false;
    for (auto& [token, counts] : _token_tag_counts) {
      if (counts.size() < label) {
        counts.push_back(0);
        requires_new_counter = true;
      } else {
        break;
      }
    }

    if (requires_new_counter) {
      _num_unique_counters++;
    }
  }

 private:
  uint32_t _number_bins;
  std::unordered_map<std::string, uint32_t> _tag_to_label;
  /*
   * multiple tags can be mapped to the same label in the model, we use only one
   * counter for each label
   */
  uint32_t _num_unique_counters;
  std::unordered_map<std::string, std::vector<uint32_t>> _token_tag_counts;

  std::unordered_map<std::string, uint32_t> _token_counts;
  uint32_t _total_tokens;
};

using TokenTagCounterPtr = std::shared_ptr<TokenTagCounter>;
}  // namespace thirdai::data::ner