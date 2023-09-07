#include "symspell.h"

SpellCheckedSentence::SpellCheckedSentence(){}

SpellCheckedSentence::SpellCheckedSentence(
    const std::vector<std::string>& tokens, const std::vector<float>& scores)
    : tokens(tokens), scores(scores) {}

SpellCheckedSentence::SpellCheckedSentence(const SpellCheckedSentence& other) : tokens(other.tokens), scores(other.scores) {}

SpellCheckedSentence SpellCheckedSentence::update_token_and_score(const std::string& token,
                                             float score,
                                             size_t index) {
    SpellCheckedSentence temp(*this);
    temp.tokens[index] = token;
    temp.scores[index] = score;
    return temp;
}

SymPreTrainer::SymPreTrainer(){}

SymPreTrainer::SymPreTrainer(int max_edit_distance,
                             bool experimental_scores, int prefix_length,
                             bool use_word_segmentation)
    : pretrainer(SymSpell(DEFAULT_INITIAL_CAPACITY, max_edit_distance, prefix_length, DEFAULT_COUNT_THRESHOLD, DEFAULT_COMPACT_LEVEL)),
      max_edit_distance(max_edit_distance),
      experimental_scores(experimental_scores),
      prefix_length(prefix_length),
      use_word_segmentation(use_word_segmentation) {
  pretrainer = SymSpell(max_edit_distance, prefix_length);
  std::cout << "Initialized a Spell Checker from scratch. Index words into "
               "the spell checker for corrections."
            << std::endl;
  this->pretrainer = pretrainer;
}

std::tuple<std::vector<std::string>, std::vector<float>>
SymPreTrainer::get_correct_spelling_single(const std::string& word,
                                           int top_k) {
  std::vector<std::string> tokens;
  std::vector<float> scores;

  std::vector<SuggestItem> results;
  if (!this->use_word_segmentation) {
    results = this->pretrainer.Lookup(
        word, Verbosity::All, max_edit_distance = this->max_edit_distance);
  } else {
    results = this->pretrainer.LookupCompound(
        word,  max_edit_distance = this->max_edit_distance);
  }

  for (SuggestItem& res : results) {
    tokens.push_back(res.term.c_str());
    if (this->experimental_scores) {
      scores.push_back(res.count * (this->max_edit_distance - res.distance));
    } else {
      scores.push_back(this->max_edit_distance - res.distance);
    }
  }

  if (top_k < static_cast<int>(tokens.size())) {
    tokens.resize(top_k);
    scores.resize(top_k);
  }

  bool found = std::find(tokens.begin(), tokens.end(), word) != tokens.end();

  if (!found) {
    if (experimental_scores) {
      if (!scores.empty()) {
        std::sort(scores.begin(), scores.end());
        int median =
            (scores.size() % 2 == 0)
                ? (scores[scores.size() / 2 - 1] + scores[scores.size() / 2]) /
                      2
                : scores[scores.size() / 2];
        scores.push_back(median);
      } else {
        scores.push_back(2);
      }
    } else {
      scores.push_back(2);
    }
  }

  return std::make_tuple(tokens, scores);
}

std::tuple<std::vector<std::vector<std::string>>,
           std::vector<std::vector<float>>>
SymPreTrainer::get_correct_spelling_list(const std::vector<std::string>& word_list,
                          int top_k) {
  std::vector<std::vector<std::string>> tokens;
  std::vector<std::vector<float>> scores;

  for (const std::string& word : word_list) {
    auto [temp_tokens, temp_scores] = get_correct_spelling_single(word, top_k);
    tokens.push_back(temp_tokens);
    scores.push_back(temp_scores);
  }

  return std::make_tuple(tokens, scores);
}

std::vector<SpellCheckedSentence>
SymPreTrainer::correct_sentence(std::vector<std::string> tokens_list,
                                int predictions_per_token,
                                int maximum_candidates, bool stop_if_found) {
  std::vector<float> scores(tokens_list.size(), 0.0F);
  SpellCheckedSentence prev = SpellCheckedSentence(tokens_list, scores);
  std::vector<SpellCheckedSentence> candidates = {prev};

  auto [predictions, prediction_scores] = this->get_correct_spelling_list(tokens_list, predictions_per_token);

  for (int i = 0; i < (int)tokens_list.size(); i++) {
    std::vector<std::string> current_candidate_tokens = predictions[i];
    std::vector<float> current_candidate_scores = prediction_scores[i];

    std::vector<SpellCheckedSentence> temp_candidates;
    for (auto candidate : candidates) {
      for (int j = 0; j < (int)current_candidate_scores.size(); j++) {
        std::string token = current_candidate_tokens[j];
        float score = current_candidate_scores[j];
        SpellCheckedSentence new_candid = candidate.update_token_and_score(token, score, i);
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
                return SpellCheckedSentence::get_score(a) >
                       SpellCheckedSentence::get_score(b);
              });
    if (temp_candidates.size() > static_cast<size_t>(maximum_candidates)) {
      temp_candidates.resize(maximum_candidates);
    }
  }
  std::vector<float> new_scores(tokens_list.size(), 1 / tokens_list.size());

  candidates.push_back(SpellCheckedSentence(tokens_list, new_scores));

  return candidates;
}

void
SymPreTrainer::index_words(
    std::vector<std::string> words_to_index,
    std::vector<int> frequency) {

  for (size_t i = 0; i < words_to_index.size(); i++) {
    const std::string& word = words_to_index[i];
    int count = frequency[i];

    // Check if the word doesn't exist in the dictionary
    if (pretrainer.Lookup(word, Verbosity::Closest, max_edit_distance = 0).size() == 0) {
      pretrainer.CreateDictionaryEntry(word, count, NULL);
    }
  }
}

// void
// SymPreTrainer::pretrain_file(const thirdai::dataset::DataSourcePtr& data) { 

//   return; 
// }
// Implement the remaining methods of SymPreTrainer as needed