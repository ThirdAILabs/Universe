#include "symspell.h"
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/udt/Defaults.h>
#include <utils/StringManipulation.h>
namespace thirdai::automl::udt {

SymPreTrainer::SymPreTrainer(uint32_t max_edit_distance, uint32_t prefix_length,
                             bool use_word_segmentation)
    : _backend(SymSpell(DEFAULT_INITIAL_CAPACITY, max_edit_distance,
                        prefix_length, DEFAULT_COUNT_THRESHOLD,
                        DEFAULT_COMPACT_LEVEL)),
      _max_edit_distance(max_edit_distance),
      _prefix_length(prefix_length),
      _use_word_segmentation(use_word_segmentation) {
  std::cout << "Initialized a Spell Checker from scratch. Index words into "
               "the spell checker for corrections."
            << std::endl;
}

std::pair<std::vector<std::string>, std::vector<float>>
SymPreTrainer::getCorrectSpellingSingle(const std::string& word,
                                        uint32_t top_k) {
  std::vector<std::string> tokens;
  std::vector<float> scores;

  std::vector<SuggestItem> results;
  if (!_use_word_segmentation) {
    results = this->_backend.Lookup(word, Verbosity::All, _max_edit_distance);
  } else {
    results = this->_backend.LookupCompound(word, _max_edit_distance);
  }

  for (const SuggestItem& res : results) {
    tokens.push_back(res.term);
    float score = _max_edit_distance - res.distance;
    scores.push_back(score);
  }

  if (top_k < tokens.size()) {
    tokens.resize(top_k);
    scores.resize(top_k);
  }

  const bool found_original_word =
      std::find(tokens.begin(), tokens.end(), word) != tokens.end();

  if (!found_original_word) {
    tokens.push_back(word);

    if (!scores.empty()) {
      std::vector<float> scores_copy;
      scores_copy.assign(scores.begin(), scores.end());
      std::sort(scores_copy.begin(), scores_copy.end());
      const float median = (scores_copy.size() % 2 == 0)
                               ? (scores_copy[scores_copy.size() / 2 - 1] +
                                  scores_copy[scores_copy.size() / 2]) /
                                     2.0F
                               : scores_copy[scores_copy.size() / 2.0F];
      scores.push_back(median);
    } else {
      scores.push_back(1.0F);
    }
  }

  return std::make_pair(tokens, scores);
}

std::pair<std::vector<std::vector<std::string>>,
          std::vector<std::vector<float>>>
SymPreTrainer::getCorrectSpellingList(const std::vector<std::string>& word_list,
                                      uint32_t top_k) {
  std::vector<std::vector<std::string>> tokens;
  std::vector<std::vector<float>> scores;

  for (const std::string& word : word_list) {
    auto [temp_tokens, temp_scores] = getCorrectSpellingSingle(word, top_k);
    tokens.push_back(temp_tokens);
    scores.push_back(temp_scores);
  }

  return std::pair(tokens, scores);
}

QueryCandidates SymPreTrainer::generateCandidates(
    const MapInputBatch& samples) {
  MapInputBatch candidate_samples;
  std::vector<uint32_t> candidate_count;

  for (const auto& input_sample : samples) {
    const std::string query_str = input_sample.begin()->second;

    const std::vector<std::string> tokenizedQuery =
        thirdai::text::tokenizeSentence(query_str);

    std::vector<SpellCheckedSentence> candidates =
        correctSentence(tokenizedQuery, defaults::PREDICTIONS_PER_TOKEN,
                        defaults::BEAM_SEARCH_WIDTH, defaults::STOP_IF_FOUND);

    for (SpellCheckedSentence& candidate : candidates) {
      MapInput sample;
      sample[input_sample.begin()->first] = candidate.getString();
      candidate_samples.push_back(sample);
    }
    candidate_count.push_back(candidates.size());
  }

  return QueryCandidates(candidate_samples, candidate_count);
}

std::vector<SpellCheckedSentence> SymPreTrainer::correctSentence(
    const std::vector<std::string>& tokens_list, uint32_t predictions_per_token,
    uint32_t maximum_candidates, bool stop_if_found) {
  const std::vector<float> scores(tokens_list.size(), 0.0F);

  const SpellCheckedSentence original =
      SpellCheckedSentence(tokens_list, scores);
  std::vector<SpellCheckedSentence> candidates = {original};

  auto [predictions, prediction_scores] =
      getCorrectSpellingList(tokens_list, predictions_per_token);

  for (size_t word_pos = 0; word_pos < tokens_list.size(); word_pos++) {
    std::vector<std::string> current_candidate_tokens = predictions[word_pos];
    std::vector<float> current_candidate_scores = prediction_scores[word_pos];

    std::vector<SpellCheckedSentence> temp_candidates;

    for (auto candidate : candidates) {
      for (size_t candidate_id = 0;
           candidate_id < current_candidate_scores.size(); candidate_id++) {
        const std::string token = current_candidate_tokens[candidate_id];
        const float score = current_candidate_scores[candidate_id];

        const SpellCheckedSentence new_candid =
            candidate.copyWithTokenAndScore(token, score, word_pos);
        temp_candidates.push_back(new_candid);
        if (stop_if_found && token == tokens_list[word_pos]) {
          break;
        }
      }
    }
    std::sort(temp_candidates.begin(), temp_candidates.end(),
              [](const SpellCheckedSentence& a, const SpellCheckedSentence& b) {
                return a.get_score() > b.get_score();
              });
    if (temp_candidates.size() > maximum_candidates) {
      temp_candidates.erase(temp_candidates.begin() + maximum_candidates,
                            temp_candidates.end());
    }
  }
  const std::vector<float> normalized_scores(
      tokens_list.size(), 1.0F / static_cast<float>(tokens_list.size()));
  candidates.push_back(SpellCheckedSentence(tokens_list, normalized_scores));

  return candidates;
}

std::pair<std::vector<uint32_t>, std::vector<float>>
SymPreTrainer::topKIdScorePairs(std::vector<std::vector<uint32_t>>& phrase_ids,
                                std::vector<std::vector<float>>& phrase_scores,
                                uint32_t top_k) {
  std::vector<std::pair<uint32_t, float>> phraseIdScorePairs;

  for (size_t i = 0; i < phrase_ids.size(); i++) {
    for (size_t j = 0; j < phrase_ids[i].size(); j++) {
      phraseIdScorePairs.emplace_back(phrase_ids[i][j], phrase_scores[i][j]);
    }
  }
  std::sort(
      phraseIdScorePairs.begin(), phraseIdScorePairs.end(),
      [](const std::pair<uint32_t, float>& a,
         const std::pair<uint32_t, float>& b) { return a.second > b.second; });
  std::vector<uint32_t> topKPhraseIds;
  std::vector<float> topKScores;

  for (size_t i = 0; i < top_k && i < phraseIdScorePairs.size(); ++i) {
    topKPhraseIds.push_back(phraseIdScorePairs[i].first);
    topKScores.push_back(phraseIdScorePairs[i].second);
  }
  return std::make_pair(topKPhraseIds, topKScores);
}

void SymPreTrainer::indexWords(
    std::unordered_map<std::string, uint32_t>& frequency_map) {
  // Optional staging object to speed up adding many entries by staging them to
  // a temporary structure.

  // initialCapacity = SYMSPELL_DICT_INITIAL_CAPACITY, The expected number of
  // words that will be added
  // Specifying an accurate initialCapacity is not neccesary,
  // but it can help speed up processing by alleviating the need for
  // data restructuring as the size grows
  SuggestionStage staging =
      SuggestionStage(defaults::SYMSPELL_DICT_INITIAL_CAPACITY);

  // Initialize an empty dictionary for _backend if pretraining first time
  std::ifstream corpusStream;
  _backend.CreateDictionary(corpusStream);

  for (const auto& [word, count] : frequency_map) {
    // Check if the word doesn't exist in the dictionary
    _backend.CreateDictionaryEntry(word, count, &staging);
  }
  _backend.CommitStaged(&staging);
}

void SymPreTrainer::pretrain(std::vector<MapInputBatch>& parsed_data) {
  std::unordered_map<std::string, uint32_t> frequency_map;

  for (const auto& batch : parsed_data) {
    for (auto input : batch) {
      const std::string line_str = input.begin()->second;

      const std::vector<std::string> tokenizedQuery =
          thirdai::text::tokenizeSentence(line_str);

      for (const std::string& token : tokenizedQuery) {
        frequency_map[token]++;
      }
    }
  }

  indexWords(frequency_map);
}

}  // namespace thirdai::automl::udt

// CEREAL_REGISTER_TYPE(thirdai::symspell::SymPreTrainer)