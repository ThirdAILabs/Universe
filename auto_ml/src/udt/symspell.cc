#include "symspell.h"

SpellCheckedSentence::SpellCheckedSentence() {}

SpellCheckedSentence::SpellCheckedSentence(
    const std::vector<std::string>& tokens, const std::vector<float>& scores)
    : tokens(tokens), scores(scores) {}

SpellCheckedSentence::SpellCheckedSentence(const SpellCheckedSentence& other)
    : tokens(other.tokens), scores(other.scores) {}

SpellCheckedSentence SpellCheckedSentence::update_token_and_score(
    const std::string& token, float score, size_t index) {
  SpellCheckedSentence temp(*this);
  temp.tokens[index] = token;
  temp.scores[index] = score;
  return temp;
}

SymPreTrainer::SymPreTrainer() {}

SymPreTrainer::SymPreTrainer(int max_edit_distance, bool experimental_scores,
                             int prefix_length, bool use_word_segmentation)
    : pretrainer(SymSpell(DEFAULT_INITIAL_CAPACITY, max_edit_distance,
                          prefix_length, DEFAULT_COUNT_THRESHOLD,
                          DEFAULT_COMPACT_LEVEL)),
      max_edit_distance(max_edit_distance),
      experimental_scores(experimental_scores),
      prefix_length(prefix_length),
      use_word_segmentation(use_word_segmentation) {
  std::cout << "Initialized a Spell Checker from scratch. Index words into "
               "the spell checker for corrections."
            << std::endl;
  this->pretrainer = pretrainer;
}

std::tuple<std::vector<std::string>, std::vector<float>>
SymPreTrainer::get_correct_spelling_single(const std::string& word, int top_k) {
  std::vector<std::string> tokens;
  std::vector<float> scores;

  std::vector<SuggestItem> results;
  if (!this->use_word_segmentation) {
    results =
        this->pretrainer.Lookup(word, Verbosity::All, this->max_edit_distance);
  } else {
    results = this->pretrainer.LookupCompound(word, this->max_edit_distance);
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
    tokens.push_back(word);
    if (experimental_scores) {
      if (!scores.empty()) {
        std::vector<int> scores_copy;
        scores_copy.assign(scores.begin(), scores.end());
        std::sort(scores_copy.begin(), scores_copy.end());
        int median = (scores_copy.size() % 2 == 0)
                         ? (scores_copy[scores_copy.size() / 2 - 1] +
                            scores_copy[scores_copy.size() / 2]) /
                               2
                         : scores_copy[scores_copy.size() / 2];
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
SymPreTrainer::get_correct_spelling_list(
    const std::vector<std::string>& word_list, int top_k) {
  std::vector<std::vector<std::string>> tokens;
  std::vector<std::vector<float>> scores;

  for (const std::string& word : word_list) {
    auto [temp_tokens, temp_scores] = get_correct_spelling_single(word, top_k);
    tokens.push_back(temp_tokens);
    scores.push_back(temp_scores);
  }

  return std::make_tuple(tokens, scores);
}

std::vector<SpellCheckedSentence> SymPreTrainer::correct_sentence(
    std::vector<std::string> tokens_list, int predictions_per_token,
    int maximum_candidates, bool stop_if_found) {
  std::vector<float> scores(tokens_list.size(), 0.0F);

  SpellCheckedSentence prev = SpellCheckedSentence(tokens_list, scores);
  std::vector<SpellCheckedSentence> candidates = {prev};

  auto [predictions, prediction_scores] =
      this->get_correct_spelling_list(tokens_list, predictions_per_token);

  for (int i = 0; i < (int)tokens_list.size(); i++) {
    std::vector<std::string> current_candidate_tokens = predictions[i];
    std::vector<float> current_candidate_scores = prediction_scores[i];

    std::vector<SpellCheckedSentence> temp_candidates;

    for (auto candidate : candidates) {
      for (uint32_t j = 0; j < current_candidate_scores.size(); j++) {
        std::string token = current_candidate_tokens[j];
        float score = current_candidate_scores[j];

        SpellCheckedSentence new_candid =
            candidate.update_token_and_score(token, score, i);
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
  std::vector<float> new_scores(tokens_list.size(),
                                1.0F / static_cast<float>(tokens_list.size()));
  candidates.push_back(SpellCheckedSentence(tokens_list, new_scores));

  return candidates;
}

void SymPreTrainer::index_words(std::vector<std::string> words_to_index,
                                std::vector<int> frequency) {
  SuggestionStage staging = SuggestionStage(16384);

  // Initialize an empty dictionary for pretrainer if pretraining first time
  std::ifstream corpusStream;
  pretrainer.CreateDictionary(corpusStream);

  for (size_t i = 0; i < words_to_index.size(); i++) {
    const std::string& word = words_to_index[i];
    int count = frequency[i];
    // Check if the word doesn't exist in the dictionary
    pretrainer.CreateDictionaryEntry(word, count, &staging);
  }
  pretrainer.CommitStaged(&staging);
}

std::vector<MapInputBatch> SymPreTrainer::parse_data(
    const DataSourcePtr& data, std::string correct_column_name,
    uint32_t batch_size) {
  std::optional<std::string> header = data->nextLine();
  if (header == std::nullopt) {
    throw std::runtime_error("File is empty.");
  }
  // Fine correct column name index
  std::stringstream headerStream(header->c_str());
  std::string columnHeader;
  std::vector<std::string> headers;
  // Parse the header to find the "target_queries" column
  while (std::getline(headerStream, columnHeader, ',')) {
    headers.push_back(columnHeader);
  }
  int targetQueriesIndex = -1;
  for (int i = 0; i < (int)headers.size(); i++) {
    if (headers[i] == correct_column_name) {
      targetQueriesIndex = i;
      break;
    }
  }
  if (targetQueriesIndex == -1) {
    throw std::runtime_error("correct queries column not found");
  }
  std::optional<std::string> line = data->nextLine();

  std::vector<MapInputBatch> parsed_data;
  MapInputBatch current_batch;

  while (line != std::nullopt) {
    std::string line_str = line->c_str();
    std::vector<std::string> comma_sep_sents;
    std::istringstream tokenStream(line_str);
    std::string token;

    while (std::getline(tokenStream, token, ',')) {
      comma_sep_sents.push_back(token);
    }
    if (targetQueriesIndex + 1 > (int)comma_sep_sents.size()) {
      line_str = "";
    }
    else{
      line_str = comma_sep_sents[targetQueriesIndex];
    }
    MapInput sample;
    sample["phrase"] = line_str;
    current_batch.push_back(sample);

    if (current_batch.size() == batch_size) {
      parsed_data.push_back(current_batch);
      current_batch.clear();
    }
    line = data->nextLine();
  }
  if (current_batch.size()) {
    parsed_data.push_back(current_batch);
    current_batch.clear();
  }
  return parsed_data;
}

void SymPreTrainer::pretrain_file(const DataSourcePtr& data,
                                  std::string correct_column_name) {
  std::unordered_map<std::string, int> frequency;

  auto parsed_data = parse_data(
      data, correct_column_name,
      thirdai::automl::udt::defaults::QUERY_REFORMULATION_BATCH_SIZE);

  for (auto batch : parsed_data)

    for (auto input : batch) {
      std::string line_str = input.begin()->second;

      std::regex word_pattern("\\b[\\w'-]+\\b");

      std::sregex_iterator word_iterator(line_str.begin(), line_str.end(),
                                         word_pattern);
      std::sregex_iterator end_iterator;
      while (word_iterator != end_iterator) {
        frequency[word_iterator->str()]++;
        ++word_iterator;
      }
    }
  std::vector<std::string> words_to_index;
  words_to_index.reserve(frequency.size());

  std::vector<int> words_frequency;
  words_frequency.reserve(frequency.size());

  for (auto kv : frequency) {
    words_to_index.push_back(kv.first);
    words_frequency.push_back(kv.second);
  }

  index_words(words_to_index, words_frequency);
  return;
}