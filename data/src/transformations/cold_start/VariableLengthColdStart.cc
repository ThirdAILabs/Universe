#include "VariableLengthColdStart.h"
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/StringConcat.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <utils/CommonChecks.h>
#include <utils/text/Stopwords.h>
#include <utils/text/StringManipulation.h>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace thirdai::data {

VariableLengthConfig::VariableLengthConfig(
    size_t covering_min_length, size_t covering_max_length,
    std::optional<uint32_t> max_covering_samples, size_t slice_min_length,
    std::optional<size_t> slice_max_length, uint32_t num_slices,
    bool add_whole_doc, bool prefilter_punctuation,
    uint32_t strong_sample_num_words, float stopword_removal_probability,
    float stopword_insertion_probability, float word_removal_probability,
    float word_perturbation_probability, size_t chars_replace_with_space,
    size_t chars_deleted, size_t chars_duplicated,
    size_t chars_replace_with_adjacents, bool nltk_tokenize)
    : covering_min_length(covering_min_length),
      covering_max_length(covering_max_length),
      max_covering_samples(max_covering_samples),
      slice_min_length(slice_min_length),
      slice_max_length(slice_max_length),
      num_slices(num_slices),
      add_whole_doc(add_whole_doc),
      prefilter_punctuation(prefilter_punctuation),
      strong_sample_num_words(strong_sample_num_words),
      stopword_removal_probability(stopword_removal_probability),
      stopword_insertion_probability(stopword_insertion_probability),
      word_removal_probability(word_removal_probability),
      word_perturbation_probability(word_perturbation_probability),
      chars_replace_with_space(chars_replace_with_space),
      chars_deleted(chars_deleted),
      chars_duplicated(chars_duplicated),
      chars_replace_with_adjacents(chars_replace_with_adjacents),
      nltk_tokenize(nltk_tokenize) {
  utils::validateGreaterThanZero(covering_min_length, "covering_min_length");
  utils::validateGreaterThanZero(covering_max_length, "covering_max_length");
  utils::validateGreaterThanZero(slice_min_length, "slice_min_length");

  if (slice_max_length) {
    utils::validateGreaterThanZero(*slice_max_length, "slice_max_length");
  }

  utils::validateBetweenZeroAndOne(stopword_removal_probability,
                                   "stopword_removal_probability");
  utils::validateBetweenZeroAndOne(stopword_insertion_probability,
                                   "stopword_insertion_probability");
  utils::validateBetweenZeroAndOne(word_removal_probability,
                                   "word_removal_probability");
  utils::validateBetweenZeroAndOne(word_perturbation_probability,
                                   "word_perturbation_probability");

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
    std::vector<std::string> weak_column_names, std::string output_column_name,
    const VariableLengthConfig& config, uint32_t seed)
    : TextAugmentationBase(std::move(strong_column_names),
                           std::move(weak_column_names),
                           std::move(output_column_name), seed),
      _config(config) {}

std::vector<std::string> VariableLengthColdStart::augmentSingleRow(
    const std::string& strong_text, const std::string& weak_text,
    uint32_t row_id_salt) const {
  // We salt with row_id to keep determinism in the augmentation while having
  // each row perturbed with a different seed.
  // We pass around this rng object to every function that has randomness (by
  // reference) for proper variability of random numbers across output
  // samples, consistency of random numbers across training runs, and for
  // performance reasons to not have to create the object over and over.
  std::mt19937 rng(_seed + row_id_salt);

  Phrase strong_phrase = convertTextToPhrase(strong_text);
  PhraseCollection phrases = getWeakPhrases(weak_text, rng);
  phrases = cold_start::mergeStrongWithWeak(
      phrases, strong_phrase, _config.strong_sample_num_words, rng);

  std::vector<std::string> output_samples;
  for (const auto& phrase : phrases) {
    std::string output_text =
        convertPhraseToText(phrase, _config.stopword_removal_probability,
                            _config.stopword_insertion_probability,
                            _config.word_removal_probability,
                            _config.word_perturbation_probability, rng);

    output_text = text::perturbCharacters(
        output_text, _config.chars_replace_with_space, _config.chars_deleted,
        _config.chars_duplicated, _config.chars_replace_with_adjacents, rng);

    if (!output_text.empty()) {
      output_samples.push_back(output_text);
    }
  }

  // if output_samples.size() <= 1 then either the weak text is too short, or
  // there is only strong text, or the sample is empty, in which case we don't
  // want to add the whole doc since we're in a degenerate case.
  if (_config.add_whole_doc && output_samples.size() > 1) {
    Phrase whole_doc_as_phrase =
        convertTextToPhrase(strong_text + " " + weak_text);

    std::string whole_doc = convertPhraseToText(
        whole_doc_as_phrase, _config.stopword_removal_probability,
        _config.stopword_insertion_probability,
        _config.word_removal_probability, _config.word_perturbation_probability,
        rng);

    whole_doc = text::perturbCharacters(
        whole_doc, _config.chars_replace_with_space, _config.chars_deleted,
        _config.chars_duplicated, _config.chars_replace_with_adjacents, rng);

    output_samples.push_back(whole_doc);
  }

  return output_samples;
}

Phrase VariableLengthColdStart::convertTextToPhrase(std::string string) const {
  if (_config.nltk_tokenize) {
    return text::custom_word_tokenize(string);
  }

  if (_config.prefilter_punctuation) {
    string = text::replacePunctuation(string, ' ');
  }
  string = text::stripWhitespace(string);

  Phrase phrase = text::tokenizeSentence(string);

  return phrase;
}

PhraseCollection VariableLengthColdStart::getWeakPhrases(
    std::string weak_text, std::mt19937& rng) const {
  Phrase weak_phrase = convertTextToPhrase(std::move(weak_text));

  if (weak_phrase.empty()) {
    return {};
  }

  PhraseCollection phrases;

  addCoveringPhrases(weak_phrase, phrases, _config.covering_min_length,
                     _config.covering_max_length, _config.max_covering_samples,
                     rng);

  addRandomSlicePhrases(weak_phrase, phrases, _config.slice_min_length,
                        _config.slice_max_length, _config.num_slices, rng);

  return phrases;
}

void VariableLengthColdStart::addCoveringPhrases(
    const Phrase& words, PhraseCollection& phrases, size_t min_len,
    size_t max_len, std::optional<size_t> max_covering_samples,
    std::mt19937& rng) {
  min_len = std::min(min_len, words.size());
  std::uniform_int_distribution<size_t> dist(min_len, max_len);

  size_t start_pos = 0;
  while (start_pos + min_len <= words.size()) {
    size_t phrase_size = std::min(dist(rng), words.size() - start_pos);
    // if there are less that min_len words left after chosing this phrase,
    // include them in this phrase by extending the phrase_size
    if (start_pos + phrase_size + min_len > words.size()) {
      phrase_size = words.size() - start_pos;
    }
    Phrase phrase(words.begin() + start_pos,
                  words.begin() + start_pos + phrase_size);
    phrases.push_back(phrase);
    start_pos += phrase_size;
  }

  if (max_covering_samples.has_value() &&
      *max_covering_samples < phrases.size()) {
    std::shuffle(phrases.begin(), phrases.end(), rng);
    phrases.resize(*max_covering_samples);
  }
}

void VariableLengthColdStart::addRandomSlicePhrases(
    const Phrase& words, PhraseCollection& phrases, size_t min_len,
    std::optional<size_t> max_len_opt, uint32_t num_slices, std::mt19937& rng) {
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

std::string VariableLengthColdStart::convertPhraseToText(
    const std::vector<std::string>& phrase, float stopword_removal_probability,
    float stopword_insertion_probability, float word_removal_probability,
    float word_perturbation_probability, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  std::string output_text;
  for (size_t i = 0; i < phrase.size(); i++) {
    std::string word = phrase[i];
    // decide to skip stopword
    if (stopword_removal_probability &&
        dist(rng) < stopword_removal_probability &&
        text::stop_words.count(word)) {
      continue;
    }

    // decide to skip the word
    if (word_removal_probability && dist(rng) < word_removal_probability) {
      continue;
    }

    // decide to perturb the word by removing either the first or last character
    float rn = dist(rng);
    if (word_perturbation_probability && rn < word_perturbation_probability) {
      if (rn < word_perturbation_probability / 2) {
        word = word.substr(1);
      } else {
        word = word.substr(0, word.length() - 1);
      }
    }

    // decide to randomly insert a stopword
    if (stopword_insertion_probability &&
        dist(rng) < stopword_insertion_probability) {
      std::string element;
      std::sample(text::stop_words.begin(), text::stop_words.end(), &element, 1,
                  rng);
      output_text.append(element);
      output_text.push_back(' ');
    }

    // add the word
    output_text.append(word);
    if (i != phrase.size() - 1) {
      output_text.push_back(' ');
    }
  }

  return output_text;
}

}  // namespace thirdai::data