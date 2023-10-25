#include "VariableLengthColdStart.h"
#include "ColdStartUtils.h"
#include <data/src/columns/ValueColumns.h>
#include <utils/StringManipulation.h>
#include <random>

namespace thirdai::data {

VariableLengthColdStart::VariableLengthColdStart(
    std::vector<std::string> strong_column_names,
    std::vector<std::string> weak_column_names, std::string label_column_name,
    std::string output_column_name, const VariableLengthConfig& config,
    uint32_t seed)
    : _strong_column_names(std::move(strong_column_names)),
      _weak_column_names(std::move(weak_column_names)),
      _label_column_name(std::move(label_column_name)),
      _output_column_name(std::move(output_column_name)),
      _covering_min_length(config.covering_min_length),
      _covering_max_length(config.covering_max_length),
      _max_covering_samples(config.max_covering_samples),
      _slice_min_length(config.slice_min_length),
      _slice_max_length(config.slice_max_length),
      _num_slices(config.num_slices),
      _add_whole_doc(config.add_whole_doc),
      _prefilter_punctuation(config.prefilter_punctuation),
      _strong_sample_num_words(config.strong_sample_num_words),
      _word_removal_probability(config.word_removal_probability),
      _seed(seed) {
  validateGreaterThanZero(_covering_min_length, "covering_min_length");
  validateGreaterThanZero(_covering_max_length, "covering_max_length");
  validateGreaterThanZero(_slice_min_length, "slice_min_length");

  if (_slice_max_length) {
    validateGreaterThanZero(*_slice_max_length, "slice_max_length");
  }

  if (_word_removal_probability < 0 or _word_removal_probability > 1.0) {
    throw std::invalid_argument(
        "word_removal_probaility must be between 0 and 1.0.");
  }

  if (_covering_min_length >= _covering_max_length) {
    throw std::invalid_argument(
        "covering_min_length must be < covering_max_length.");
  }

  if (_slice_max_length.has_value() &&
      _slice_min_length >= *_slice_max_length) {
    throw std::invalid_argument("slice_min_length must be < slice_max_length.");
  }
}

void VariableLengthColdStart::validateGreaterThanZero(
    uint32_t parameter, const std::string& parameter_name) {
  if (parameter <= 0) {
    throw std::invalid_argument("Invalid length parameter: " + parameter_name +
                                " must be greater than 0.");
  }
}

ColumnMap VariableLengthColdStart::apply(ColumnMap columns,
                                         State& state) const {
  (void)state;

  auto label_column = columns.getValueColumn<std::string>(_label_column_name);

  std::vector<std::string> augmented_labels;
  std::vector<std::string> augmented_data;

  std::exception_ptr exception = nullptr;

#pragma omp parallel for default(none) \
    shared(label_column, columns, augmented_data, augmented_labels, exception)
  for (uint64_t row_id = 0; row_id < label_column->numRows(); row_id++) {
    try {
      std::string labels = label_column->value(row_id);

      std::string weak_text = cold_start::concatenateStringColumnEntries(
          columns, row_id, _weak_column_names, /* delimiter= */ " ");

      std::string strong_text = cold_start::concatenateStringColumnEntries(
          columns, row_id, _strong_column_names, /* delimiter= */ " ");

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

  // Shuffle the augmented data and augmented labels (in the same order).
  // We have to use std::shuffle and two RNGs from <random> with the same state
  //  for reasons described here: https://stackoverflow.com/a/16969267
  std::mt19937 rng_1(_seed);
  auto rng_2 = rng_1;

  std::shuffle(augmented_data.begin(), augmented_data.end(), rng_1);
  std::shuffle(augmented_labels.begin(), augmented_labels.end(), rng_2);

  auto augmented_label_column =
      ValueColumn<std::string>::make(std::move(augmented_labels));

  auto augmented_data_column =
      ValueColumn<std::string>::make(std::move(augmented_data));

  std::unordered_map<std::string, ColumnPtr> new_columns;
  new_columns.emplace(_label_column_name, augmented_label_column);
  new_columns.emplace(_output_column_name, augmented_data_column);
  ColumnMap augmented_column_map(new_columns);
  return augmented_column_map;
}

std::vector<std::string> VariableLengthColdStart::augmentSingleRow(
    std::string& strong_text, std::string& weak_text) const {
  std::vector<std::string> strong_phrase =
      cold_start::getStrongPhrase(strong_text);
  std::cout << "STRONG PHRASE: " << std::endl;
  for (auto word : strong_phrase) {
    std::cout << word << " ";
  }
  std::cout << std::endl;
  std::vector<std::vector<std::string>> phrases = getWeakPhrases(weak_text);
  std::cout << "WEAK PHRASES: " << std::endl;
  for (auto phrase : phrases) {
    for (auto word : phrase) {
      std::cout << word << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  cold_start::mergeStrongWithWeak(phrases, strong_phrase,
                                  _strong_sample_num_words, _seed);
  std::cout << "MERGED PHRASES: " << std::endl;
  for (auto phrase : phrases) {
    for (auto word : phrase) {
      std::cout << word << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::mt19937 rng(_seed);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  std::vector<std::string> output_samples;
  for (const auto& phrase : phrases) {
    std::string output_text;
    for (const auto& word : phrase) {
      if (_word_removal_probability == 0 ||
          dist(rng) > _word_removal_probability) {
        output_text.append(word);
        output_text.append(" ");
      }
    }
    output_samples.push_back(output_text);
  }

  if (_add_whole_doc) {
    output_samples.push_back(strong_text + " " + weak_text);
  }

  return output_samples;
}

std::vector<std::vector<std::string>> VariableLengthColdStart::getWeakPhrases(
    std::string& weak_text) const {
  if (_prefilter_punctuation) {
    text::replacePunctuationWithSpaces(weak_text);
  }

  std::vector<std::string> words = cold_start::splitByWhitespace(weak_text);

  std::vector<std::vector<std::string>> phrases;

  addCoveringPhrases(words, phrases, _covering_min_length, _covering_max_length,
                     _max_covering_samples, _seed);

  std::cout << "COVERING PHRASES: " << std::endl;
  for (auto phrase : phrases) {
    for (auto word : phrase) {
      std::cout << word << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  addRandomSlicePhrases(words, phrases, _slice_min_length, _slice_max_length,
                        _num_slices, _seed);

  std::cout << "SLICE PHRASES: " << std::endl;
  for (auto phrase : phrases) {
    for (auto word : phrase) {
      std::cout << word << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return phrases;
}

void VariableLengthColdStart::addCoveringPhrases(
    const std::vector<std::string>& words,
    std::vector<std::vector<std::string>>& phrases, uint32_t min_len,
    uint32_t max_len, std::optional<uint32_t> max_covering_samples,
    uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> dist(min_len, max_len);

  size_t start_pos = 0;
  while (start_pos + min_len <= words.size()) {
    int phrase_size =
        std::min(static_cast<size_t>(dist(rng)), words.size() - start_pos);
    if (start_pos + phrase_size + min_len > words.size()) {
      phrase_size += words.size() - phrase_size - start_pos;
    }
    std::vector<std::string> phrase(words.begin() + start_pos,
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
    const std::vector<std::string>& words,
    std::vector<std::vector<std::string>>& phrases, uint32_t min_len,
    std::optional<uint32_t> max_len_opt, uint32_t num_slices, uint32_t seed) {
  std::mt19937 rng(seed);
  uint32_t max_len = max_len_opt.has_value()
                         ? std::min<uint32_t>(words.size(), *max_len_opt)
                         : words.size();
  std::uniform_int_distribution<uint32_t> len_dist(min_len, max_len);

  for (uint32_t i = 0; i < num_slices; i++) {
    uint32_t len = len_dist(rng);
    std::uniform_int_distribution<uint32_t> start_pos_dist(0,
                                                           words.size() - len);
    uint32_t start_pos = start_pos_dist(rng);
    std::vector<std::string> phrase(words.begin() + start_pos,
                                    words.begin() + start_pos + len);
    phrases.push_back(phrase);
  }
}

}  // namespace thirdai::data