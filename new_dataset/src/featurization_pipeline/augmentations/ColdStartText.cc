#include "ColdStartText.h"
#include <new_dataset/src/featurization_pipeline/Column.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::data {

ColdStartTextAugmentation::ColdStartTextAugmentation(
    std::vector<std::string> strong_column_names,
    std::vector<std::string> weak_column_names, std::string label_column_name,
    std::string output_column_name, std::optional<uint32_t> weak_min_len,
    std::optional<uint32_t> weak_max_len,
    std::optional<uint32_t> weak_chunk_len,
    std::optional<uint32_t> weak_sample_num_words, uint32_t weak_sample_reps,
    std::optional<uint32_t> strong_max_len,
    std::optional<uint32_t> strong_sample_num_words, uint32_t seed)
    : _strong_column_names(std::move(strong_column_names)),
      _weak_column_names(std::move(weak_column_names)),
      _label_column_name(std::move(label_column_name)),
      _output_column_name(std::move(output_column_name)),
      _weak_min_len(weak_min_len),
      _weak_max_len(weak_max_len),
      _weak_chunk_len(weak_chunk_len),
      _weak_sample_num_words(weak_sample_num_words),
      _weak_sample_reps(weak_sample_reps),
      _strong_max_len(strong_max_len),
      _strong_sample_num_words(strong_sample_num_words),
      _seed(seed) {
  // Validate input parameters.
  validateGreaterThanZero(_weak_min_len, "weak_min_len");
  validateGreaterThanZero(_weak_max_len, "weak_max_len");
  validateGreaterThanZero(_weak_chunk_len, "weak_chunk_len");
  validateGreaterThanZero(_weak_sample_num_words, "weak_sample_num_words");
  validateGreaterThanZero(_strong_max_len, "strong_max_len");
  validateGreaterThanZero(_strong_sample_num_words, "strong_sample_num_words");

  if (weak_sample_reps <= 0) {
    throw std::invalid_argument(
        "Invalid parameter: weak_sample_reps "
        "must be greater than 0.");
  }
  if (weak_sample_reps > 1000) {
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
    std::optional<uint32_t> parameter, const std::string parameter_name) {
  if (parameter && parameter.value() <= 0) {
    std::string error_message = "Invalid length parameter: ";
    error_message = error_message + parameter_name;
    error_message = error_message + "must be greater than 0.";
    throw std::invalid_argument(error_message);
  }
}

ColumnMap ColdStartTextAugmentation::apply(const ColumnMap& columns) {
  auto label_column = columns.getTokenArrayColumn(_label_column_name);
  std::vector<std::vector<uint32_t>> augmented_labels;
  std::vector<std::string> augmented_data;

  for (uint64_t row_id = 0; row_id < label_column->numRows(); row_id++) {
    std::string weak_text = concatenateStringColumnEntries(
        columns, row_id, _weak_column_names, ". ");

    std::string strong_text = concatenateStringColumnEntries(
        columns, row_id, _strong_column_names, " ");
    // Now that we have both the weak and strong text, pass them into the
    // phrase generation pipeline to self-supervised (label, phrase) pairs.
    Phrase strong_phrase = getStrongPhrase(strong_text);
    PhraseCollection phrases = getWeakPhrases(weak_text);
    mergeStrongWithWeak(phrases, strong_phrase);

    std::vector<uint32_t> labels;
    for (auto& label : (*label_column)[row_id]) {
      labels.push_back(label);
    }
    for (auto& phrase : phrases) {
      // Add (label, phrase) to the output data.
      std::string output_text = "";
      for (auto& word : phrase) {
        output_text.append(word);
        output_text.append(" ");
      }
      augmented_data.push_back(output_text);
      augmented_labels.push_back(labels);
    }
  }
  // Shuffle the augmented data and augmented labels (in the same order).
  // We have to use std::shuffle and two RNGs from <random> with the same state
  //  for reasons described here: https://stackoverflow.com/a/16969267
  std::mt19937 rng_1(_seed);
  auto rng_2 = rng_1;

  std::shuffle(augmented_data.begin(), augmented_data.end(), rng_1);
  std::shuffle(augmented_labels.begin(), augmented_labels.end(), rng_2);

  // Finally, make a new ColumnMap (i.e. dataset) out of the augmented data and
  // augmented labels. Note that the VectorSparseArrayColumn constructor takes
  // an optional uint32_t dimension instead of an optional DimensionInfo.
  std::optional<columns::DimensionInfo> label_dimension_info =
      label_column->dimension();
  std::optional<uint32_t> label_dimension = std::nullopt;
  if (label_dimension_info) {
    label_dimension = label_dimension_info.value().dim;
  }

  columns::TokenArrayColumnPtr augmented_label_column =
      std::make_shared<columns::CppTokenArrayColumn>(augmented_labels,
                                                     label_dimension);
  columns::StringColumnPtr augmented_data_column =
      std::make_shared<columns::CppStringColumn>(augmented_data);

  std::unordered_map<std::string, columns::ColumnPtr> new_columns;
  new_columns.emplace(_label_column_name, augmented_label_column);
  new_columns.emplace(_output_column_name, augmented_data_column);
  ColumnMap augmented_column_map(new_columns);
  return augmented_column_map;
}

void ColdStartTextAugmentation::replacePunctuationWithSpaces(std::string& s) {
  std::replace_if(
      s.begin(), s.end(), [](const char c) -> bool { return std::ispunct(c); },
      ' ');
}

void ColdStartTextAugmentation::stripWhitespace(std::string& s) {
  auto first_valid = s.find_first_not_of(" \t\f\v\n\r");
  auto last_valid = s.find_last_not_of(" \t\f\v\n\r");
  if (first_valid == std::string::npos || last_valid == std::string::npos) {
    // Whole string is whitespace.
    s = "";
  } else {
    s = s.substr(first_valid, last_valid + 1 - first_valid);
  }
}

ColdStartTextAugmentation::Phrase ColdStartTextAugmentation::getStrongPhrase(
    std::string& s) {
  replacePunctuationWithSpaces(s);
  stripWhitespace(s);
  Phrase strong_phrase = splitByWhitespace(s);
  if (_strong_max_len) {
    if (strong_phrase.size() > _strong_max_len.value()) {
      strong_phrase.resize(_strong_max_len.value());
    }
  }
  return strong_phrase;
}

ColdStartTextAugmentation::Phrase ColdStartTextAugmentation::splitByWhitespace(
    std::string& s) {
  ColdStartTextAugmentation::Phrase phrase;
  std::string word;
  std::istringstream s_stream(s);
  while (s_stream >> word) {
    phrase.push_back(word);
  }
  return phrase;
}

ColdStartTextAugmentation::PhraseCollection
ColdStartTextAugmentation::getWeakPhrases(std::string& s) {
  std::string::iterator phrase_start;
  std::string::iterator phrase_end;
  phrase_start = s.begin();
  phrase_end = s.begin();

  ColdStartTextAugmentation::PhraseCollection phrases;
  // The natural phrases are not necessarily long enough or short enough
  // on their own. We may have to cut or concatenate them to get phrases
  // of the desired length. We do this in a single pass by storing
  // intermediate results in the following phrase accumulators.
  ColdStartTextAugmentation::Phrase current_natural_phrase;
  ColdStartTextAugmentation::Phrase current_chunk_phrase;

  while (phrase_end != s.end()) {
    phrase_end = std::find_if(phrase_end, s.end(), [](const char c) -> bool {
      return std::ispunct(c);
    });
    std::string natural_phrase_text(phrase_start, phrase_end);
    replacePunctuationWithSpaces(natural_phrase_text);
    stripWhitespace(natural_phrase_text);
    phrase_start = phrase_end;
    if (phrase_end != s.end()) {
      ++phrase_end;  // Necessary to not re-find the same punctuation again.
    }
    if (natural_phrase_text.size() == 0) continue;
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
  if (current_natural_phrase.size() > 0) {
    phrases.push_back(current_natural_phrase);
  }
  if (current_chunk_phrase.size() > 0) {
    phrases.push_back(current_chunk_phrase);
  }
  if (_weak_sample_num_words) {
    phrases = sampleFromPhrases(
        /* phrases= */ phrases,
        /* num_to_sample= */ _weak_sample_num_words.value(),
        /* num_reps= */ _weak_sample_reps);
  }
  return phrases;
}

ColdStartTextAugmentation::PhraseCollection
ColdStartTextAugmentation::sampleFromPhrases(
    ColdStartTextAugmentation::PhraseCollection& phrases,
    uint32_t num_to_sample, uint32_t num_reps) {
  // Only iterate over the original phrases, as we append new ones to the end.
  if (num_reps == 0) {
    throw std::invalid_argument(
        "Invalid number of sampling repetitions: "
        "must be greater than 0.");
  }
  PhraseCollection output_phrases;
  std::mt19937 rng(_seed);
  for (auto& phrase : phrases) {
    if (phrase.size() > num_to_sample) {
      // Then we can downsample some sub-phrases.
      std::vector<uint32_t> permutation(phrase.size());
      std::iota(permutation.begin(), permutation.end(), 0);
      for (uint32_t rep = 0; rep < num_reps; rep++) {
        std::shuffle(permutation.begin(), permutation.end(), rng);
        std::sort(permutation.begin(), permutation.begin() + num_to_sample);
        Phrase new_phrase;
        for (uint32_t j = 0; j < num_to_sample; j++) {
          new_phrase.push_back(phrase[permutation[j]]);
        }
        output_phrases.push_back(new_phrase);
      }
    } else {
      // there are not enough words in the phrase to choose num_to_sample.
      output_phrases.push_back(phrase);
    }
  }
  return output_phrases;
}

std::string ColdStartTextAugmentation::concatenateStringColumnEntries(
    const ColumnMap& columns, const uint64_t row_num,
    const std::vector<std::string>& column_names, const std::string delimiter) {
  std::string output_text = "";
  for (auto& column_name : column_names) {
    auto column = columns.getStringColumn(column_name);
    output_text.append((*column)[row_num]);
    output_text.append(delimiter);
  }
  return output_text;
}

void ColdStartTextAugmentation::mergeStrongWithWeak(
    ColdStartTextAugmentation::PhraseCollection& weak_phrases,
    Phrase& strong_phrase) {
  ColdStartTextAugmentation::PhraseCollection downsampled_strong_phrases;
  if (_strong_sample_num_words) {
    // If we have to sample from the strong phrase, we create N independently
    // sampled sub-strings, where N is the number of weak phrases that we have.
    downsampled_strong_phrases.push_back(strong_phrase);
    downsampled_strong_phrases = sampleFromPhrases(
        /* phrases= */ downsampled_strong_phrases,
        /* num_to_sample= */ _strong_sample_num_words.value(),
        /* num_reps= */ weak_phrases.size());
  }
  for (uint32_t i = 0; i < weak_phrases.size(); i++) {
    ColdStartTextAugmentation::Phrase phrase_to_concatenate;
    if (downsampled_strong_phrases.size() > i) {
      phrase_to_concatenate = downsampled_strong_phrases[i];
    } else {
      // This can happen if we don't downsample the strong phrase, but also
      // if the strong phrase is too short to downsample to the desired length.
      phrase_to_concatenate = strong_phrase;
    }
    uint32_t original_size = weak_phrases[i].size();
    for (auto& word : phrase_to_concatenate) {
      weak_phrases[i].push_back(word);
    }
    // Make the strong phrase come at the start instead of the end.
    std::rotate(weak_phrases[i].begin(),
                weak_phrases[i].begin() + original_size, weak_phrases[i].end());
  }
}

}  // namespace thirdai::data
