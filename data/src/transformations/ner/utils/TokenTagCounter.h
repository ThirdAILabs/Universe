#pragma once
#include <archive/src/Archive.h>
#include <data/src/transformations/Tabular.h>
#include <data/src/transformations/ner/utils/TagTracker.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace thirdai::data::ner {
class TokenTagCounter {
 public:
  TokenTagCounter(uint32_t number_bins, utils::TagTrackerPtr tag_tracker);

  void addTokenTag(const std::string& token, const std::string& tag);

  std::string getTokenEncoding(const std::string& token) const;

  explicit TokenTagCounter(const ar::Archive& archive);

  ar::ConstArchivePtr toArchive() const;

  void addNewCounter() {
    for (auto& [token, counts] : _token_tag_counts) {
      counts.push_back(0);
    }
  }

  void setTagTracker(utils::TagTrackerPtr tag_tracker) {
    _tag_tracker = std::move(tag_tracker);
  }

 private:
  //  number of bins to divide the interval [0,1] to discretize the float ratio
  uint32_t _number_bins;

  utils::TagTrackerPtr _tag_tracker;
  /*
   * multiple tags can be mapped to the same label in the model, we use only one
   * counter for each label
   */
  std::unordered_map<std::string, std::vector<uint32_t>> _token_tag_counts;

  std::unordered_map<std::string, uint32_t> _token_counts;
  uint32_t _total_tokens;
};

using TokenTagCounterPtr = std::shared_ptr<TokenTagCounter>;
}  // namespace thirdai::data::ner