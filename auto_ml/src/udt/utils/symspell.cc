#include "symspell.h"
#include <utils/StringManipulation.h>

namespace thirdai::symspell {

SpellCheckedSentence::SpellCheckedSentence(
    const std::vector<std::string>& tokens, const std::vector<float>& scores)
    : _tokens(tokens), _scores(scores) {}

SpellCheckedSentence::SpellCheckedSentence(const SpellCheckedSentence& other)
    : _tokens(other._tokens), _scores(other._scores) {}

SpellCheckedSentence SpellCheckedSentence::updateTokenAndScore(
    const std::string& token, float score, size_t index) {
  SpellCheckedSentence temp(*this);
  temp._tokens[index] = token;
  temp._scores[index] = score;
  return temp;
}

SymPreTrainer::SymPreTrainer(uint32_t max_edit_distance,
                             bool experimental_scores, uint32_t prefix_length,
                             bool use_word_segmentation)
    : _backend(SymSpell(DEFAULT_INITIAL_CAPACITY, max_edit_distance,
                        prefix_length, DEFAULT_COUNT_THRESHOLD,
                        DEFAULT_COMPACT_LEVEL)),
      _max_edit_distance(max_edit_distance),
      _experimental_scores(experimental_scores),
      _prefix_length(prefix_length),
      _use_word_segmentation(use_word_segmentation) {
  std::cout
      << "Initialized a Spell Checker from scratch. Index words uint32_to "
         "the spell checker for corrections."
      << std::endl;
  this->_backend = _backend;
}

std::pair<std::vector<std::string>, std::vector<float>>
SymPreTrainer::get_correct_spelling_single(const std::string& word,
                                           uint32_t top_k) {
  std::vector<std::string> tokens;
  std::vector<float> scores;

  std::vector<SuggestItem> results;
  if (!_use_word_segmentation) {
    results = this->_backend.Lookup(word, Verbosity::All, _max_edit_distance);
  } else {
    results = this->_backend.LookupCompound(word, _max_edit_distance);
  }

  for (SuggestItem& res : results) {
    tokens.push_back(res.term.c_str());
    if (_experimental_scores) {
      scores.push_back(res.count * (_max_edit_distance - res.distance));
    } else {
      scores.push_back(_max_edit_distance - res.distance);
    }
  }

  if (top_k < static_cast<uint32_t>(tokens.size())) {
    tokens.resize(top_k);
    scores.resize(top_k);
  }

  bool found = std::find(tokens.begin(), tokens.end(), word) != tokens.end();

  if (!found) {
    tokens.push_back(word);
    if (_experimental_scores) {
      if (!scores.empty()) {
        std::vector<float> scores_copy;
        scores_copy.assign(scores.begin(), scores.end());
        std::sort(scores_copy.begin(), scores_copy.end());
        float median = (scores_copy.size() % 2 == 0)
                           ? (scores_copy[scores_copy.size() / 2 - 1] +
                              scores_copy[scores_copy.size() / 2]) /
                                 2.0F
                           : scores_copy[scores_copy.size() / 2.0F];
        scores.push_back(median);
      } else {
        scores.push_back(2.0F);
      }
    } else {
      scores.push_back(2.0F);
    }
  }

  return std::make_pair(tokens, scores);
}

std::pair<std::vector<std::vector<std::string>>,
          std::vector<std::vector<float>>>
SymPreTrainer::get_correct_spelling_list(
    const std::vector<std::string>& word_list, uint32_t top_k) {
  std::vector<std::vector<std::string>> tokens;
  std::vector<std::vector<float>> scores;

  for (const std::string& word : word_list) {
    auto [temp_tokens, temp_scores] = get_correct_spelling_single(word, top_k);
    tokens.push_back(temp_tokens);
    scores.push_back(temp_scores);
  }

  return std::pair(tokens, scores);
}

std::pair<MapInputBatch, std::vector<uint32_t>>
SymPreTrainer::generate_candidates(const MapInputBatch& samples) {
  MapInputBatch candidate_samples;
  std::vector<uint32_t> candidate_count;

  for (auto input_sample : samples) {
    std::string query_str = input_sample.begin()->second;

    std::vector<std::string> tokenizedQuery =
        thirdai::text::tokenizeSentence(query_str);

    std::vector<SpellCheckedSentence> candidates =
        correct_sentence(tokenizedQuery, PREDICTIONS_PER_TOKEN,
                         BEAM_SEARCH_WIDTH, STOP_IF_FOUND);

    for (SpellCheckedSentence& candidate : candidates) {
      MapInput sample;
      sample[input_sample.begin()->first] = candidate.get_string();
      candidate_samples.push_back(sample);
    }
    candidate_count.push_back(candidates.size());
  }

  return make_pair(candidate_samples, candidate_count);
}

std::vector<SpellCheckedSentence> SymPreTrainer::correct_sentence(
    std::vector<std::string> tokens_list, uint32_t predictions_per_token,
    uint32_t maximum_candidates, bool stop_if_found) {
  std::vector<float> scores(tokens_list.size(), 0.0F);

  SpellCheckedSentence prev = SpellCheckedSentence(tokens_list, scores);
  std::vector<SpellCheckedSentence> candidates = {prev};

  auto [predictions, prediction_scores] =
      this->get_correct_spelling_list(tokens_list, predictions_per_token);

  for (size_t i = 0; i < tokens_list.size(); i++) {
    std::vector<std::string> current_candidate_tokens = predictions[i];
    std::vector<float> current_candidate_scores = prediction_scores[i];

    std::vector<SpellCheckedSentence> temp_candidates;

    for (auto candidate : candidates) {
      for (size_t j = 0; j < current_candidate_scores.size(); j++) {
        std::string token = current_candidate_tokens[j];
        float score = current_candidate_scores[j];

        SpellCheckedSentence new_candid =
            candidate.updateTokenAndScore(token, score, i);
        temp_candidates.push_back(new_candid);
        if (stop_if_found) {
          if (token == tokens_list[i]) {
            break;
          }
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
  std::vector<float> new_scores(tokens_list.size(),
                                1.0F / static_cast<float>(tokens_list.size()));
  candidates.push_back(SpellCheckedSentence(tokens_list, new_scores));

  return candidates;
}

std::pair<std::vector<uint32_t>, std::vector<float>>
SymPreTrainer::accumulate_scores(std::vector<std::vector<uint32_t>> phrase_ids,
                                 std::vector<std::vector<float>> phrase_scores,
                                 std::optional<uint32_t> top_k) {
  std::vector<uint32_t> flattenedPhraseIds;
  for (const auto& vec : phrase_ids) {
    flattenedPhraseIds.insert(flattenedPhraseIds.end(), vec.begin(), vec.end());
  }

  std::vector<float> flattenedScores;
  for (const auto& vec : phrase_scores) {
    flattenedScores.insert(flattenedScores.end(), vec.begin(), vec.end());
  }
  std::vector<std::pair<uint32_t, float>> phraseIdScorePairs;

  for (size_t i = 0; i < flattenedPhraseIds.size(); ++i) {
    phraseIdScorePairs.emplace_back(flattenedPhraseIds[i], flattenedScores[i]);
  }
  std::sort(
      phraseIdScorePairs.begin(), phraseIdScorePairs.end(),
      [](const std::pair<uint32_t, float>& a,
         const std::pair<uint32_t, float>& b) { return a.second > b.second; });
  std::vector<uint32_t> topKPhraseIds;
  std::vector<float> topKScores;

  for (size_t i = 0; i < top_k.value() && i < phraseIdScorePairs.size(); ++i) {
    topKPhraseIds.push_back(phraseIdScorePairs[i].first);
    topKScores.push_back(phraseIdScorePairs[i].second);
  }
  return std::make_pair(topKPhraseIds, topKScores);
}

void SymPreTrainer::index_words(std::vector<std::string> words_to_index,
                                std::vector<uint32_t> frequency) {
  // Optional staging object to speed up adding many entries by staging them to
  // a temporary structure.

  // initialCapacity = 16384, The expected number of words that will be
  // added.</param>
  //<remarks>Specifying ann accurate initialCapacity is not essential,
  // but it can help speed up processing by alleviating the need for
  // data restructuring as the size grows.</remarks>
  SuggestionStage staging = SuggestionStage(16384);

  // Initialize an empty dictionary for _backend if pretraining first time
  std::ifstream corpusStream;
  _backend.CreateDictionary(corpusStream);

  for (size_t i = 0; i < words_to_index.size(); i++) {
    const std::string& word = words_to_index[i];
    uint32_t count = frequency[i];
    // Check if the word doesn't exist in the dictionary
    _backend.CreateDictionaryEntry(word, count, &staging);
  }
  _backend.CommitStaged(&staging);
}

void SymPreTrainer::pretrain_file(std::vector<MapInputBatch> parsed_data) {
  std::unordered_map<std::string, uint32_t> frequency;

  for (auto batch : parsed_data)

    for (auto input : batch) {
      std::string line_str = input.begin()->second;

      std::vector<std::string> tokenizedQuery =
          thirdai::text::tokenizeSentence(line_str);

      for (std::string token : tokenizedQuery) {
        frequency[token]++;
      }
    }

  std::vector<std::string> words_to_index;
  words_to_index.reserve(frequency.size());

  std::vector<uint32_t> words_frequency;
  words_frequency.reserve(frequency.size());

  for (auto kv : frequency) {
    words_to_index.push_back(kv.first);
    words_frequency.push_back(kv.second);
  }

  index_words(words_to_index, words_frequency);
  return;
}

}  // namespace thirdai::symspell