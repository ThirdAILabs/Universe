#include "VariableLengthColdStart.h"
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/StringConcat.h>
#include <utils/CommonChecks.h>
#include <utils/StringManipulation.h>
#include <random>

namespace thirdai::data {

VariableLengthConfig::VariableLengthConfig(
    size_t covering_min_length, size_t covering_max_length,
    std::optional<uint32_t> max_covering_samples, size_t slice_min_length,
    std::optional<size_t> slice_max_length, uint32_t num_slices,
    bool add_whole_doc, bool prefilter_punctuation,
    uint32_t strong_sample_num_words, float word_removal_probability)
    : covering_min_length(covering_min_length),
      covering_max_length(covering_max_length),
      max_covering_samples(max_covering_samples),
      slice_min_length(slice_min_length),
      slice_max_length(slice_max_length),
      num_slices(num_slices),
      add_whole_doc(add_whole_doc),
      prefilter_punctuation(prefilter_punctuation),
      strong_sample_num_words(strong_sample_num_words),
      word_removal_probability(word_removal_probability) {
  utils::validateGreaterThanZero(covering_min_length, "covering_min_length");
  utils::validateGreaterThanZero(covering_max_length, "covering_max_length");
  utils::validateGreaterThanZero(slice_min_length, "slice_min_length");

  if (slice_max_length) {
    utils::validateGreaterThanZero(*slice_max_length, "slice_max_length");
  }

  if (word_removal_probability < 0 or word_removal_probability > 1.0) {
    throw std::invalid_argument(
        "word_removal_probaility must be between 0 and 1.0.");
  }

  if (covering_min_length > covering_max_length) {
    throw std::invalid_argument(
        "covering_min_length must be <= covering_max_length.");
  }

  if (slice_max_length.has_value() && slice_min_length > *slice_max_length) {
    throw std::invalid_argument(
        "slice_min_length must be <= slice_max_length.");
  }
}

VariableLengthColdStart::VariableLengthColdStart(
    std::vector<std::string> strong_column_names,
    std::vector<std::string> weak_column_names, std::string label_column_name,
    std::string output_column_name, const VariableLengthConfig& config,
    uint32_t seed)
    : TextAugmentationBase(
          std::move(strong_column_names), std::move(weak_column_names),
          std::move(label_column_name), std::move(output_column_name), seed),
      _config(config) {}

std::vector<std::string> VariableLengthColdStart::augmentSingleRow(
    const std::string& strong_text, const std::string& weak_text) const {
  Phrase strong_phrase = cold_start::getStrongPhrase(strong_text);
  PhraseCollection phrases = getWeakPhrases(weak_text);
  phrases = cold_start::mergeStrongWithWeak(
      phrases, strong_phrase, _config.strong_sample_num_words, _seed);

  std::mt19937 rng(_seed);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  std::vector<std::string> output_samples;
  for (const auto& phrase : phrases) {
    std::string output_text;
    for (const auto& word : phrase) {
      if (_config.word_removal_probability == 0 ||
          dist(rng) > _config.word_removal_probability) {
        output_text.append(word);
        output_text.push_back(' ');
      }
    }

    if (!output_text.empty()) {
      output_samples.push_back(output_text);
    }
  }

  // if output_samples.size() <= 1 then either the weak text is too short, or
  // there is only strong text, or the sample is empty, in which case we don't
  // want to add the whole doc since we're in a degenerate case.
  if (_config.add_whole_doc && output_samples.size() > 1) {
    std::string whole_doc = strong_text + " " + weak_text;
    if (_config.prefilter_punctuation) {
      whole_doc = text::replacePunctuation(whole_doc, ' ');
    }
    output_samples.push_back(whole_doc);
  }

  return output_samples;
}

PhraseCollection VariableLengthColdStart::getWeakPhrases(
    std::string weak_text) const {
  if (_config.prefilter_punctuation) {
    weak_text = text::replacePunctuation(weak_text, ' ');
  }

  Phrase weak_phrase = text::tokenizeSentence(weak_text);

  if (weak_phrase.empty()) {
    return {};
  }

  PhraseCollection phrases;

  addCoveringPhrases(weak_phrase, phrases, _config.covering_min_length,
                     _config.covering_max_length, _config.max_covering_samples,
                     _seed);

  addRandomSlicePhrases(weak_phrase, phrases, _config.slice_min_length,
                        _config.slice_max_length, _config.num_slices, _seed);

  return phrases;
}

void VariableLengthColdStart::addCoveringPhrases(
    const Phrase& words, PhraseCollection& phrases, size_t min_len,
    size_t max_len, std::optional<size_t> max_covering_samples, uint32_t seed) {
  std::mt19937 rng(seed);
  min_len = std::min(min_len, words.size());
  std::uniform_int_distribution<size_t> dist(min_len, max_len);

  size_t start_pos = 0;
  while (start_pos + min_len <= words.size()) {
    size_t phrase_size = std::min(dist(rng), words.size() - start_pos);
    // if there are less that min_len words left after chosing this phrase,
    // include them in this phrase by extending the phrase_size
    if (start_pos + phrase_size + min_len > words.size()) {
      phrase_size += words.size() - phrase_size - start_pos;
    }
    Phrase phrase(words.begin() + start_pos,
                  words.begin() + start_pos + phrase_size);
    phrases.push_back(phrase);
    start_pos += phrase_size;
  }

  if (max_covering_samples.has_value() &&
      *max_covering_samples < phrases.size()) {
    std::shuffle(phrases.begin(), phrases.end(), std::mt19937{seed});
    phrases.resize(*max_covering_samples);
  }
}

void VariableLengthColdStart::addRandomSlicePhrases(
    const Phrase& words, PhraseCollection& phrases, size_t min_len,
    std::optional<size_t> max_len_opt, uint32_t num_slices, uint32_t seed) {
  std::mt19937 rng(seed);
  min_len = std::min(min_len, words.size());
  size_t max_len = max_len_opt.has_value()
                       ? std::min(words.size(), *max_len_opt)
                       : words.size();
  std::uniform_int_distribution<size_t> len_dist(min_len, max_len);

  for (uint32_t i = 0; i < num_slices; i++) {
    size_t len = len_dist(rng);
    std::uniform_int_distribution<size_t> start_pos_dist(0, words.size() - len);
    size_t start_pos = start_pos_dist(rng);
    Phrase phrase(words.begin() + start_pos, words.begin() + start_pos + len);
    phrases.push_back(phrase);
  }
}

}  // namespace thirdai::data