#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/udt/Defaults.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <utils/src/SymSpellCpp/SymSpell.h>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using MapInputBatch = thirdai::dataset::MapInputBatch;
using MapInput = thirdai::dataset::MapInput;

namespace thirdai::automl::udt {
struct QueryCandidates {
 private:
  std::vector<uint32_t> offsets;
  MapInputBatch candidate_batches;

 public:
  explicit QueryCandidates(MapInputBatch& candidates,
                           std::vector<uint32_t> candidate_count)
      : offsets(std::move(candidate_count)), candidate_batches(candidates) {
    for (size_t i = 1; i < offsets.size(); i++) {
      offsets[i] += offsets[i - 1];
    }
  }

  // Get range of candidates for query_id
  std::pair<uint32_t, uint32_t> getRange(uint32_t query_id) const {
    uint32_t begin_offset = (query_id == 0) ? 0 : offsets[query_id - 1];
    uint32_t end_offset = offsets[query_id];
    return std::make_pair(begin_offset, end_offset);
  }

  MapInputBatch getSample() { return candidate_batches; }
};

class SpellCheckedSentence {
 private:
  std::vector<std::string> _tokens;
  std::vector<float> _scores;

 public:
  SpellCheckedSentence(std::vector<std::string> tokens,
                       std::vector<float> scores)
      : _tokens(std::move(tokens)), _scores(std::move(scores)) {}

  SpellCheckedSentence copyWithTokenAndScore(const std::string& token,
                                             float score, uint32_t index) {
    SpellCheckedSentence temp(*this);
    temp._tokens[index] = token;
    temp._scores[index] = score;
    return temp;
  }

  float getScore() const {
    float total_score = 0.0F;
    for (const float score : _scores) {
      total_score += score;
    }
    return total_score;
  }

  std::string getString() {
    std::string result = std::accumulate(
        _tokens.begin(), _tokens.end(), std::string(),
        [](const std::string& a, const std::string& b) { return a + b + " "; });
    result.pop_back();  // remove last space
    return result;
  }
};

class SymPreTrainer {
 public:
  // WordSegmentation divides a string into words by inserting missing spaces at
  // the appropriate positions misspelled words are corrected and do not affect
  // segmentation existing spaces are allowed and considered for optimum
  // segmentation

  /// >Length of prefix, from which internal dictionary of symspell are
  /// generated.

  // To know more about the variables, refer SymSpell.h in SymSpellCPP repo

  SymPreTrainer(uint32_t max_edit_distance, uint32_t prefix_length,
                bool use_word_segmentation);

  QueryCandidates generateCandidates(const MapInputBatch& samples);

  static std::pair<std::vector<uint32_t>, std::vector<float>> topKIdScorePairs(
      std::vector<std::vector<uint32_t>>& phrase_ids,
      std::vector<std::vector<float>>& phrase_scores, uint32_t top_k);

  std::pair<std::vector<std::string>, std::vector<float>>
  getCorrectSpellingSingle(const std::string& word, uint32_t top_k);

  void pretrain(std::vector<MapInputBatch>& parsed_data);

 private:
  SymSpell _backend;
  uint32_t _max_edit_distance;
  uint32_t _prefix_length;
  bool _use_word_segmentation;

  SymPreTrainer(){};

  void indexWords(std::unordered_map<std::string, uint32_t>& frequency_map);

  std::pair<std::vector<std::vector<std::string>>,
            std::vector<std::vector<float>>>
  getCorrectSpellingList(const std::vector<std::string>& word_list,
                         uint32_t top_k);

  std::vector<SpellCheckedSentence> correctSentence(
      const std::vector<std::string>& tokens_list,
      uint32_t predictions_per_token, uint32_t maximum_candidates,
      bool stop_if_found);

  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& archive) {
    archive(_backend, _max_edit_distance, _prefix_length,
            _use_word_segmentation);
  }
};

using SymSpellPtr = std::shared_ptr<SymPreTrainer>;

}  // namespace thirdai::automl::udt
