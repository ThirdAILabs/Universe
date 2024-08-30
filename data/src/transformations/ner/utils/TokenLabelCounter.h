#pragma once
#include <archive/src/Archive.h>
#include <data/src/transformations/Tabular.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace thirdai::data::ner {
class TokenLabelCounter {
 public:
  TokenLabelCounter(uint32_t number_bins, uint32_t num_labels);

  void addTokenLabel(const std::string& token, uint32_t label);

  std::string getTokenEncoding(const std::string& token) const;

  explicit TokenLabelCounter(const ar::Archive& archive);

  ar::ConstArchivePtr toArchive() const;

  void addNewCounter() {
    for (auto& [token, counts] : _token_label_counts) {
      counts.push_back(0);
    }
    _num_labels++;
  }

 private:
  //  number of bins to divide the interval [0,1] to discretize the float ratio
  uint32_t _number_bins;

  // total different labels : [0,n)
  uint32_t _num_labels;

  // token -> frequency count vector for labels
  std::unordered_map<std::string, std::vector<uint32_t>> _token_label_counts;

  // token -> cumulative token frequency
  std::unordered_map<std::string, uint32_t> _token_counts;

  // total number of unique tokens seen
  uint32_t _total_tokens;
};

using TokenLabelCounterPtr = std::shared_ptr<TokenLabelCounter>;
}  // namespace thirdai::data::ner