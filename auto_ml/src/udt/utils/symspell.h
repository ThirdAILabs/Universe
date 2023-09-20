#pragma once

#include <SymspellCPP/include/SymSpell.h>
#include <auto_ml/src/udt/Defaults.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace thirdai::dataset;
using namespace thirdai::automl::udt::defaults;

namespace thirdai::symspell {

class SpellCheckedSentence {
 private:
  std::vector<std::string> _tokens;
  std::vector<float> _scores;

 public:
  SpellCheckedSentence(const std::vector<std::string>& tokens,
                       const std::vector<float>& scores);

  SpellCheckedSentence(const SpellCheckedSentence& other);

  SpellCheckedSentence updateTokenAndScore(const std::string& token,
                                           float score, size_t index);

  float get_score() const {
    float total_score = 0.0F;
    for (float score : _scores) {
      total_score += score;
    }
    return total_score;
  }

  std::string get_string() {
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

  /// <summary>Length of prefix, from which internal dictionary of symspell are
  /// generated.</summary>

  // To know more about the variables, refer SymSpell.h in SymSpellCPP repo

  SymPreTrainer(uint32_t max_edit_distance, bool experimental_scores,
                uint32_t prefix_length, bool use_word_segmentation);

  std::pair<std::vector<std::string>, std::vector<float>>
  get_correct_spelling_single(const std::string& word, uint32_t top_k);

  std::pair<std::vector<std::vector<std::string>>,
            std::vector<std::vector<float>>>
  get_correct_spelling_list(const std::vector<std::string>& word_list,
                            uint32_t top_k);

  std::pair<MapInputBatch, std::vector<uint32_t>> generate_candidates(
      const MapInputBatch& samples);

  std::vector<SpellCheckedSentence> correct_sentence(
      std::vector<std::string> tokens_list, uint32_t predictions_per_token,
      uint32_t maximum_candidates, bool stop_if_found);

  void index_words(std::vector<std::string> words_to_index,
                   std::vector<uint32_t> frequency);

  std::pair<std::vector<uint32_t>, std::vector<float>> accumulate_scores(
      std::vector<std::vector<uint32_t>> phrase_ids,
      std::vector<std::vector<float>> phrase_scores,
      std::optional<uint32_t> top_k);

  void pretrain_file(std::vector<MapInputBatch> parsed_data);

 private:
  SymSpell _backend;
  uint32_t _max_edit_distance;
  bool _experimental_scores;
  uint32_t _prefix_length;
  bool _use_word_segmentation;

  SymPreTrainer(){};
};

using SymSpellPtr = std::shared_ptr<SymPreTrainer>;

}  // namespace thirdai::symspell
