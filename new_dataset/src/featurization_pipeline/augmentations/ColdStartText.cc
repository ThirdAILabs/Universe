#include "ColdStartText.h"

#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
#include <new_dataset/src/featurization_pipeline/Column.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>


namespace thirdai::dataset {

ColumnMap ColdStartTextAugmentation::apply(const ColumnMap& columns) {
  auto label_column = columns.getSparseArrayColumn(_label_column_name);
  std::vector<std::vector<uint32_t>> augmented_labels;
  std::vector<std::string> augmented_data;

  for (uint64_t row_id = 0; row_id < label_column->numRows(); row_id++){
    std::string weak_text = "";
    for (auto& weak_name : _weak_column_names){
      auto weak_column = columns.getStringColumn(weak_name);
      weak_text.append((*weak_column)[row_id]);
      weak_text.append(". ");
    }
    std::string strong_text = "";
    for (auto& strong_name : _strong_column_names){
      auto strong_column = columns.getStringColumn(strong_name);
      strong_text.append((*strong_column)[row_id]);
      strong_text.append(" ");
    }
    // Now that we have both the weak and strong text, pass them into the
    // phrase generation pipeline to get self-supervised (label, phrase) pairs.
    replacePunctuation(strong_text);
    stripWhitespace(strong_text);
    Phrase_t strong_phrase = splitByWhitespace(strong_text);
    if (_strong_max_len){
      if (strong_phrase.size() > _strong_max_len.value()){
        strong_phrase.resize(_strong_max_len.value());
      }
    }
    PhraseCollection_t phrases = getPhrases(weak_text);
    if (_weak_downsample_num){
      sampleFromPhrases(phrases, _weak_downsample_num.value(), _weak_downsample_reps);
    }
    mergeStrongWithWeak(phrases, strong_phrase);

    std::vector<uint32_t> labels;
    for (auto& label : (*label_column)[row_id]){
      labels.push_back(label);
    }
    for (auto& phrase : phrases){
      // Add (label, phrase) to the output data.
      std::string output_text = "";
      for (auto& word : phrase){
        output_text.append(word);
        output_text.append(" ");
      }
      augmented_data.push_back(output_text);
      augmented_labels.push_back(labels);
    }
  }
  // Finally, make a new ColumnMap (i.e. dataset) out of the augmented data and
  // augmented labels. Note that the VectorSparseArrayColumn constructor takes
  // an optional uint32_t dimension instead of an optional DimensionInfo.
  std::optional<DimensionInfo> label_dimension_info = label_column->dimension();
  std::optional<uint32_t> label_dimension = std::nullopt;
  if (label_dimension_info){ label_dimension = label_dimension_info.value().dim; }

  SparseArrayColumnPtr augmented_label_column =
    std::make_shared<VectorSparseArrayColumn>(augmented_labels, label_dimension);
  StringColumnPtr augmented_data_column = 
    std::make_shared<VectorStringValueColumn>(augmented_data);
  
  std::unordered_map<std::string, ColumnPtr> new_columns;
  new_columns.emplace(_label_column_name, augmented_label_column);
  new_columns.emplace(_output_column_name, augmented_data_column);
  ColumnMap augmented_column_map(new_columns);
  return augmented_column_map;
}


void ColdStartTextAugmentation::replacePunctuation(std::string& s){
  std::replace_if(s.begin(), s.end(), [](const char c) -> bool {
    return std::ispunct(c);}, ' ');
}


void ColdStartTextAugmentation::stripWhitespace(std::string& s){
  auto first_valid = s.find_first_not_of(" \t\f\v\n\r");
  auto last_valid = s.find_last_not_of(" \t\f\v\n\r");
  if (first_valid == std::string::npos || last_valid == std::string::npos){
    // Whole string is whitespace.
    s = "";
  } else {
    s = s.substr(first_valid, last_valid+1-first_valid);
  }
}


ColdStartTextAugmentation::Phrase_t ColdStartTextAugmentation::splitByWhitespace(std::string& s){
  ColdStartTextAugmentation::Phrase_t phrase;
  std::string word;
  std::istringstream s_stream(s);
  while (s_stream >> word){
    phrase.push_back(word);
  }
  return phrase;
}


ColdStartTextAugmentation::PhraseCollection_t ColdStartTextAugmentation::getPhrases(std::string& s){
  std::string::iterator phrase_start;
  std::string::iterator phrase_end;
  phrase_start = s.begin();
  phrase_end = s.begin();

  ColdStartTextAugmentation::PhraseCollection_t phrases;
  // The natural phrases are not necessarily long enough or short enough
  // on their own. We may have to cut or concatenate them to get phrases
  // of the desired length. We do this in a single pass by storing
  // intermediate results in the following phrase accumulators.
  ColdStartTextAugmentation::Phrase_t current_natural_phrase;
  ColdStartTextAugmentation::Phrase_t current_chunk_phrase;

  while (phrase_end != s.end()){
    phrase_end = std::find_if(phrase_end, s.end(), [] (const char c) -> bool {
      return std::ispunct(c);});
    std::string natural_phrase_text(phrase_start, phrase_end);
    replacePunctuation(natural_phrase_text);
    stripWhitespace(natural_phrase_text);
    phrase_start = phrase_end;
    if (phrase_end != s.end()){
      ++phrase_end;  // Necessary to not re-find the same punctuation again.
    }
    if (natural_phrase_text.size() == 0) continue;
    // Next, iterate through all words in the phrase.
    std::string word;
    std::istringstream phrase_stream(natural_phrase_text);
    while (phrase_stream >> word){
      current_natural_phrase.push_back(word);
      if (_weak_max_len){
        // We specified a maximum length for natural phrases, so we must
        // verify that current_natural_phrase does not exceed this length.
        if (current_natural_phrase.size() >= _weak_max_len.value()){
          // Add phrase and reset the accumulator.
          phrases.push_back(current_natural_phrase);
          current_natural_phrase.clear();
        }
      }
      if (_weak_chunk_len){
        // We specified a length for non-naturally delimited, chunked phrases.
        current_chunk_phrase.push_back(word);
        if (current_chunk_phrase.size() >= _weak_chunk_len.value()
            && _weak_chunk_len.value() > 0){
          // Add phrase and reset the accumulator.
          phrases.push_back(current_chunk_phrase);
          current_chunk_phrase.clear();
        }
      }
    }
    if (_weak_min_len){
      if (current_natural_phrase.size() >= _weak_min_len.value()){
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
  if (current_natural_phrase.size() > 0){
    phrases.push_back(current_natural_phrase);
  }
  if (current_chunk_phrase.size() > 0){
    phrases.push_back(current_chunk_phrase);
  }
  return phrases;
}


void ColdStartTextAugmentation::sampleFromPhrases(
    ColdStartTextAugmentation::PhraseCollection_t &phrases, uint32_t num_to_sample, uint32_t num_reps){
  // Only iterate over the original phrases, as we append new ones to the end.
  if (num_reps == 0){ num_reps = 1; } // Rather than throw error, just fix it.
  PhraseCollection_t output_phrases;
  std::mt19937 rng(_seed);
  for (auto& phrase : phrases){
    if (phrase.size() > num_to_sample){
      // Then we can downsample some sub-phrases.
      std::vector<uint32_t> permutation(phrase.size());
      std::iota(permutation.begin(), permutation.end(), 0);
      for (uint32_t rep = 0; rep < num_reps; rep++){
        std::shuffle(permutation.begin(), permutation.end(), rng);
        std::sort(permutation.begin(), permutation.begin()+num_to_sample);
        Phrase_t new_phrase;
        for (uint32_t j = 0; j < num_to_sample; j++){
          new_phrase.push_back(phrase[permutation[j]]);
        }
        output_phrases.push_back(new_phrase);
      }
    } else {
      // there are not enough words in the phrase to choose num_to_sample.
      output_phrases.push_back(phrase);
    }
  }
  phrases = output_phrases;
}


void ColdStartTextAugmentation::mergeStrongWithWeak(
    ColdStartTextAugmentation::PhraseCollection_t &weak_phrases, Phrase_t &strong_phrase){
  // If we are asked to sample from the strong phrase, we do it. Otherwise, we
  // just concatenate the same strong phrase with every weak phrase.
  ColdStartTextAugmentation::PhraseCollection_t downsampled_strong_phrases;
  if (_strong_downsample_num){
    downsampled_strong_phrases.push_back(strong_phrase);
    sampleFromPhrases(downsampled_strong_phrases,
                      _strong_downsample_num.value(),
                      weak_phrases.size());
  }
  for (uint32_t i = 0; i < weak_phrases.size(); i++){
    ColdStartTextAugmentation::Phrase_t phrase_to_concatenate;
    if (downsampled_strong_phrases.size() > i){
      phrase_to_concatenate = downsampled_strong_phrases[i];
    } else {
      // This happens when we don't downsample the strong phrase, but also when
      // the strong phrase is too short to downsample to the desired length.
      phrase_to_concatenate = strong_phrase;
    }
    uint32_t original_size = weak_phrases[i].size();
    for (auto& word : phrase_to_concatenate){
      weak_phrases[i].push_back(word);
    }
    // Make the strong phrase come at the start instead of the end.
    std::rotate(weak_phrases[i].begin(),
                weak_phrases[i].begin() + original_size,
                weak_phrases[i].end());
  }
}


}  // namespace thirdai::dataset
