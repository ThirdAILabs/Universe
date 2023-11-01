#include "ColdStartUtils.h"
#include <utils/StringManipulation.h>
#include <random>

namespace thirdai::data::cold_start {

void mergeStrongWithWeak(PhraseCollection& weak_phrases, Phrase& strong_phrase,
                         std::optional<uint32_t> strong_sample_num_words,
                         uint32_t seed) {
  if (weak_phrases.empty()) {
    // TODO(any) evaluate alternatives for if we have no weak phrases. Maybe
    // sampling from the title?
    weak_phrases = {strong_phrase};
    return;
  }

  PhraseCollection downsampled_strong_phrases;
  if (strong_sample_num_words) {
    // If we have to sample from the strong phrase, we create N independently
    // sampled sub-strings, where N is the number of weak phrases that we have.
    downsampled_strong_phrases.push_back(strong_phrase);
    downsampled_strong_phrases = sampleFromPhrases(
        /* phrases= */ downsampled_strong_phrases,
        /* num_to_sample= */ strong_sample_num_words.value(),
        /* num_reps= */ weak_phrases.size(), seed);
  }
  for (uint32_t i = 0; i < weak_phrases.size(); i++) {
    Phrase phrase_to_concatenate;
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

PhraseCollection sampleFromPhrases(const PhraseCollection& phrases,
                                   uint32_t num_to_sample, uint32_t num_reps,
                                   uint32_t seed) {
  // Only iterate over the original phrases, as we append new ones to the end.
  if (num_reps == 0) {
    throw std::invalid_argument(
        "Invalid number of sampling repetitions: "
        "must be greater than 0.");
  }
  PhraseCollection output_phrases;
  std::mt19937 rng(seed);
  for (const auto& phrase : phrases) {
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

Phrase getStrongPhrase(const std::string& strong_text_in,
                       std::optional<uint32_t> max_len) {
  std::string strong_text = text::replacePunctuationWithSpaces(strong_text_in);
  strong_text = text::stripWhitespace(strong_text);
  Phrase strong_phrase = text::tokenizeSentence(strong_text);
  if (max_len) {
    if (strong_phrase.size() > max_len.value()) {
      strong_phrase.resize(max_len.value());
    }
  }
  return strong_phrase;
}

}  // namespace thirdai::data::cold_start