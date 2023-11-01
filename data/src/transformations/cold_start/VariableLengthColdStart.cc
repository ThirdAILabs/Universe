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
    uint32_t strong_sample_num_words, float word_removal_probability,
    uint32_t seed)
    : covering_min_length(covering_min_length),
      covering_max_length(covering_max_length),
      max_covering_samples(max_covering_samples),
      slice_min_length(slice_min_length),
      slice_max_length(slice_max_length),
      num_slices(num_slices),
      add_whole_doc(add_whole_doc),
      prefilter_punctuation(prefilter_punctuation),
      strong_sample_num_words(strong_sample_num_words),
      word_removal_probability(word_removal_probability),
      seed(seed) {
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
    std::string output_column_name, const VariableLengthConfig& config)
    : _strong_column_names(std::move(strong_column_names)),
      _weak_column_names(std::move(weak_column_names)),
      _label_column_name(std::move(label_column_name)),
      _output_column_name(std::move(output_column_name)),
      _config(config) {}

ColumnMap VariableLengthColdStart::apply(ColumnMap columns,
                                         State& state) const {
  (void)state;

  auto label_column = columns.getValueColumn<std::string>(_label_column_name);

  auto strong_concat_transform = StringConcat(
      _strong_column_names, /* output_column_name= */ "strong_text",
      /* separator= */ ". ");
  auto weak_concat_transform =
      StringConcat(_weak_column_names, /* output_column_name= */ "weak_text",
                   /* separator= */ " ");
  columns = strong_concat_transform.apply(columns, state);
  columns = weak_concat_transform.apply(columns, state);
  auto strong_column = columns.getValueColumn<std::string>("strong_text");
  auto weak_column = columns.getValueColumn<std::string>("weak_text");

  std::vector<std::string> augmented_labels;
  std::vector<std::string> augmented_data;

  std::exception_ptr exception = nullptr;

#pragma omp parallel for default(none)                               \
    shared(label_column, strong_column, weak_column, augmented_data, \
           augmented_labels, exception)
  for (uint64_t row_id = 0; row_id < label_column->numRows(); row_id++) {
    try {
      std::string labels = label_column->value(row_id);
      std::string strong_text = strong_column->value(row_id);
      std::string weak_text = weak_column->value(row_id);

      std::vector<std::string> augmented_samples =
          augmentSingleRow(strong_text, weak_text);

#pragma omp critical
      {
        for (auto& sample : augmented_samples) {
          if (!sample.empty()) {
            augmented_data.emplace_back(std::move(sample));
            augmented_labels.push_back(labels);
          }
        }
      }
    } catch (std::exception& e) {
#pragma omp critical
      exception = std::current_exception();
    }
  }

  if (exception) {
    std::rethrow_exception(exception);
  }

  auto augmented_label_column =
      ValueColumn<std::string>::make(std::move(augmented_labels));

  auto augmented_data_column =
      ValueColumn<std::string>::make(std::move(augmented_data));

  std::unordered_map<std::string, ColumnPtr> new_columns;
  new_columns.emplace(_label_column_name, augmented_label_column);
  new_columns.emplace(_output_column_name, augmented_data_column);
  ColumnMap augmented_column_map(new_columns);
  augmented_column_map.shuffle(_config.seed);
  return augmented_column_map;
}

std::vector<std::string> VariableLengthColdStart::augmentSingleRow(
    const std::string& strong_text, const std::string& weak_text) const {
  Phrase strong_phrase = cold_start::getStrongPhrase(strong_text);
  PhraseCollection phrases = getWeakPhrases(weak_text);
  cold_start::mergeStrongWithWeak(
      phrases, strong_phrase, _config.strong_sample_num_words, _config.seed);

  std::mt19937 rng(_config.seed);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  std::vector<std::string> output_samples;
  for (const auto& phrase : phrases) {
    std::string output_text;
    for (const auto& word : phrase) {
      if (_config.word_removal_probability == 0 ||
          dist(rng) > _config.word_removal_probability) {
        output_text.append(word);
        output_text.append(" ");
      }
    }

    if (!output_text.empty()) {
      output_samples.push_back(output_text);
    }
  }

  // if output_samples.size() < 1 then either the weak text is too short, or
  // there is only strong text, or the sample is empty, in which case we don't
  // want to add the whole doc since we're in a degenerate case.
  if (_config.add_whole_doc && output_samples.size() > 1) {
    std::string whole_doc = strong_text + " " + weak_text;
    if (_config.prefilter_punctuation) {
      whole_doc = text::replacePunctuationWithSpaces(whole_doc);
    }
    output_samples.push_back(whole_doc);
  }

  return output_samples;
}

PhraseCollection VariableLengthColdStart::getWeakPhrases(
    std::string weak_text) const {
  if (_config.prefilter_punctuation) {
    weak_text = text::replacePunctuationWithSpaces(weak_text);
  }

  Phrase weak_phrase = text::tokenizeSentence(weak_text);

  if (weak_phrase.empty()) {
    return {};
  }

  PhraseCollection phrases;

  addCoveringPhrases(weak_phrase, phrases, _config.covering_min_length,
                     _config.covering_max_length, _config.max_covering_samples,
                     _config.seed);

  addRandomSlicePhrases(weak_phrase, phrases, _config.slice_min_length,
                        _config.slice_max_length, _config.num_slices,
                        _config.seed);

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