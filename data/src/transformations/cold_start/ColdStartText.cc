#include "ColdStartText.h"
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/StringConcat.h>
#include <utils/StringManipulation.h>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::data {

ColdStartTextAugmentation::ColdStartTextAugmentation(
    std::vector<std::string> strong_column_names,
    std::vector<std::string> weak_column_names, std::string output_column_name,
    const ColdStartConfig& config, uint32_t seed)
    : TextAugmentationBase(std::move(strong_column_names),
                           std::move(weak_column_names),
                           std::move(output_column_name), seed),
      _weak_min_len(config.weak_min_len),
      _weak_max_len(config.weak_max_len),
      _weak_chunk_len(config.weak_chunk_len),
      _weak_sample_num_words(config.weak_sample_num_words),
      _weak_sample_reps(config.weak_sample_reps),
      _strong_max_len(config.strong_max_len),
      _strong_sample_num_words(config.strong_sample_num_words) {
  // Validate input parameters.
  validateGreaterThanZero(_weak_min_len, "weak_min_len");
  validateGreaterThanZero(_weak_max_len, "weak_max_len");
  validateGreaterThanZero(_weak_chunk_len, "weak_chunk_len");
  validateGreaterThanZero(_weak_sample_num_words, "weak_sample_num_words");
  validateGreaterThanZero(_strong_max_len, "strong_max_len");
  validateGreaterThanZero(_strong_sample_num_words, "strong_sample_num_words");

  if (_weak_sample_reps <= 0) {
    throw std::invalid_argument(
        "Invalid parameter: weak_sample_reps "
        "must be greater than 0.");
  }
  if (_weak_sample_reps > 1000) {
    throw std::invalid_argument(
        "Invalid parameter: weak_sample_reps "
        "should be smaller than 1000");
  }
  if (_weak_min_len && _weak_max_len &&
      _weak_min_len.value() > _weak_max_len.value()) {
    throw std::invalid_argument(
        "Invalid length parameter: weak_min_len "
        "must not be greater than weak_max_len.");
  }
  // If strong_sample_num_words is bigger than strong_max_len, then the
  // sampling algorithm in sampleFromPhrases will do nothing since each
  // strong phrase will be too short.
  if (_strong_sample_num_words && _strong_max_len &&
      _strong_sample_num_words.value() >= _strong_max_len.value()) {
    throw std::invalid_argument(
        "Invalid parameter: strong_sample_num_words "
        "must be less than strong_max_len.");
  }
  // We could perform a similar check for natural / chunked weak phrases, but
  // there are situations where a user may specify a value of
  // weak_sample_num_words that will affect the chunk phrases but not natural
  // phrases (or vice versa) and it is difficult to validate all combinations.
}

void ColdStartTextAugmentation::validateGreaterThanZero(
    std::optional<uint32_t> parameter, const std::string& parameter_name) {
  if (parameter && parameter.value() <= 0) {
    std::string error_message = "Invalid length parameter: ";
    error_message = error_message + parameter_name;
    error_message = error_message + "must be greater than 0.";
    throw std::invalid_argument(error_message);
  }
}

std::vector<std::string> ColdStartTextAugmentation::augmentSingleRow(
    const std::string& strong_text, const std::string& weak_text) const {
  // Now that we have both the weak and strong text, pass them into the
  // phrase generation pipeline to self-supervised (label, phrase) pairs.
  Phrase strong_phrase =
      cold_start::getStrongPhrase(strong_text, _strong_max_len);
  PhraseCollection phrases = getWeakPhrases(weak_text);
  phrases = cold_start::mergeStrongWithWeak(phrases, strong_phrase,
                                            _strong_sample_num_words, _seed);

  std::vector<std::string> output_samples;
  for (const auto& phrase : phrases) {
    // Add (label, phrase) to the output data.
    std::string output_text;
    for (const auto& word : phrase) {
      output_text.append(word);
      output_text.push_back(' ');
    }
    output_samples.push_back(output_text);
  }

  return output_samples;
}

PhraseCollection ColdStartTextAugmentation::getWeakPhrases(
    std::string s) const {
  std::string::iterator phrase_start;
  std::string::iterator phrase_end;
  phrase_start = s.begin();
  phrase_end = s.begin();

  PhraseCollection phrases;
  // The natural phrases are not necessarily long enough or short enough
  // on their own. We may have to cut or concatenate them to get phrases
  // of the desired length. We do this in a single pass by storing
  // intermediate results in the following phrase accumulators.
  Phrase current_natural_phrase;
  Phrase current_chunk_phrase;

  while (phrase_end != s.end()) {
    phrase_end = std::find_if(phrase_end, s.end(), [](const char c) -> bool {
      return std::ispunct(c);
    });
    std::string natural_phrase_text(phrase_start, phrase_end);
    natural_phrase_text = text::replacePunctuation(natural_phrase_text, ' ');
    natural_phrase_text = text::stripWhitespace(natural_phrase_text);
    phrase_start = phrase_end;
    if (phrase_end != s.end()) {
      ++phrase_end;  // Necessary to not re-find the same punctuation again.
    }
    if (natural_phrase_text.empty()) {
      continue;
    }
    // Next, iterate through all words in the phrase.
    std::string word;
    std::istringstream phrase_stream(natural_phrase_text);
    while (phrase_stream >> word) {
      current_natural_phrase.push_back(word);
      if (_weak_max_len &&
          current_natural_phrase.size() >= _weak_max_len.value()) {
        phrases.push_back(current_natural_phrase);
        current_natural_phrase.clear();
      }
      if (_weak_chunk_len) {
        current_chunk_phrase.push_back(word);
        if (current_chunk_phrase.size() >= _weak_chunk_len.value()) {
          // Note that this accumulator does not reset on punctuation.
          phrases.push_back(current_chunk_phrase);
          current_chunk_phrase.clear();
        }
      }
    }
    if (_weak_min_len) {
      if (current_natural_phrase.size() >= _weak_min_len.value()) {
        phrases.push_back(current_natural_phrase);
        current_natural_phrase.clear();
      }
      // If natural phrase wasn't long enough to qualify, we leave it in the
      // accumulator to concatenate it with the next phrase.
    } else {
      // We did not specify a minimum natural phrase length, so we can add it.
      phrases.push_back(current_natural_phrase);
      current_natural_phrase.clear();
    }
  }
  // Add any in-progress phrases. This also acts as a final fallback in case we
  // did not add any phrases at all yet.
  if (!current_natural_phrase.empty()) {
    phrases.push_back(current_natural_phrase);
  }
  if (!current_chunk_phrase.empty()) {
    phrases.push_back(current_chunk_phrase);
  }
  if (_weak_sample_num_words) {
    phrases = cold_start::sampleFromPhrases(
        /* phrases= */ phrases,
        /* words_per_sampled_phrase= */ _weak_sample_num_words.value(),
        /* n_sampled_phrases= */ _weak_sample_reps, _seed);
  }
  return phrases;
}

std::vector<std::string> ColdStartTextAugmentation::augmentMapInput(
    const automl::MapInput& document) {
  std::string strong_text;
  for (const auto& strong_col : _strong_column_names) {
    if (!document.count(strong_col)) {
      throw std::invalid_argument(
          "Strong column not found in the provided document.");
    }
    strong_text.append(document.at(strong_col));
    strong_text.append(" ");
  }
  std::string weak_text;
  for (const auto& weak_col : _weak_column_names) {
    if (!document.count(weak_col)) {
      throw std::invalid_argument(
          "Weak column not found in the provided document.");
    }
    weak_text.append(document.at(weak_col));
    weak_text.append(". ");
  }

  return augmentSingleRow(strong_text, weak_text);
}

}  // namespace thirdai::data
