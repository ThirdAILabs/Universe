#include "TextAugmentationUtils.h"
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/StringConcat.h>
#include <utils/StringManipulation.h>
#include <random>

namespace thirdai::data::cold_start {

TextAugmentationBase::TextAugmentationBase(
    std::vector<std::string> strong_column_names,
    std::vector<std::string> weak_column_names, std::string label_column_name,
    std::string output_column_name, uint32_t seed)
    : _strong_column_names(std::move(strong_column_names)),
      _weak_column_names(std::move(weak_column_names)),
      _label_column_name(std::move(label_column_name)),
      _output_column_name(std::move(output_column_name)),
      _seed(seed) {}

ColumnMap TextAugmentationBase::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto label_column = columns.getValueColumn<std::string>(_label_column_name);

  // Note: The original cold start implementation used a simple concatenation
  // function that appended an extra delimiter at the end. The StringConcat
  // transformation does not do this. This should not have any major affect on
  // cold start, leaving this note just in case anyone is investigating any
  // differences.
  auto strong_concat_transform = StringConcat(
      _strong_column_names, /* output_column_name= */ "strong_text",
      /* separator= */ " ");
  auto weak_concat_transform =
      StringConcat(_weak_column_names, /* output_column_name= */ "weak_text",
                   /* separator= */ ". ");
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

  // TODO (any) Should we use the column map permutation method to augment all
  // the rows that are not the data? that way we could apply the cold start
  // augmentation when we have more than just text and a label?
  std::unordered_map<std::string, ColumnPtr> new_columns;
  new_columns.emplace(_label_column_name, augmented_label_column);
  new_columns.emplace(_output_column_name, augmented_data_column);
  ColumnMap augmented_column_map(new_columns);
  return augmented_column_map;
}

PhraseCollection mergeStrongWithWeak(
    const PhraseCollection& weak_phrases, const Phrase& strong_phrase,
    std::optional<uint32_t> strong_sample_num_words, uint32_t seed) {
  if (weak_phrases.empty()) {
    return {strong_phrase};
  }

  PhraseCollection downsampled_strong_phrases;
  if (strong_sample_num_words) {
    // If we have to sample from the strong phrase, we create N independently
    // sampled sub-strings, where N is the number of weak phrases that we have.
    downsampled_strong_phrases = sampleFromPhrases(
        /* phrases= */ {strong_phrase},
        /* words_per_sampled_phrase= */ strong_sample_num_words.value(),
        /* n_sampled_phrases= */ weak_phrases.size(), seed);
  }

  PhraseCollection output_phrases(weak_phrases.size());
  for (uint32_t i = 0; i < weak_phrases.size(); i++) {
    Phrase concat_phrase;
    if (downsampled_strong_phrases.size() > i) {
      concat_phrase = downsampled_strong_phrases[i];
    } else {
      concat_phrase = strong_phrase;
    }
    concat_phrase.insert(concat_phrase.end(),
                         std::make_move_iterator(weak_phrases[i].begin()),
                         std::make_move_iterator(weak_phrases[i].end()));
    output_phrases[i] = std::move(concat_phrase);
  }
  return output_phrases;
}

PhraseCollection sampleFromPhrases(const PhraseCollection& phrases,
                                   uint32_t words_per_sampled_phrase,
                                   uint32_t n_sampled_phrases, uint32_t seed) {
  // Only iterate over the original phrases, as we append new ones to the end.
  if (n_sampled_phrases == 0) {
    throw std::invalid_argument(
        "Invalid number of sampling repetitions: "
        "must be greater than 0.");
  }
  PhraseCollection output_phrases;
  std::mt19937 rng(seed);
  for (const auto& phrase : phrases) {
    if (phrase.size() > words_per_sampled_phrase) {
      // Then we can downsample some sub-phrases.
      std::vector<uint32_t> permutation(phrase.size());
      std::iota(permutation.begin(), permutation.end(), 0);
      for (uint32_t rep = 0; rep < n_sampled_phrases; rep++) {
        std::shuffle(permutation.begin(), permutation.end(), rng);
        std::sort(permutation.begin(),
                  permutation.begin() + words_per_sampled_phrase);
        Phrase new_phrase;
        for (uint32_t j = 0; j < words_per_sampled_phrase; j++) {
          new_phrase.push_back(phrase[permutation[j]]);
        }
        output_phrases.push_back(new_phrase);
      }
    } else {
      // there are not enough words in the phrase to choose
      output_phrases.push_back(phrase);
    }
  }
  return output_phrases;
}

Phrase getStrongPhrase(const std::string& strong_text_in,
                       std::optional<uint32_t> max_len) {
  std::string strong_text = text::replacePunctuation(strong_text_in, ' ');
  strong_text = text::stripWhitespace(strong_text);

  // Note: This is slightly different than the original cold start
  // implementation. This tokenization/split function splits on any character
  // that isn't alpha-numeric. The old version just split on whitespace. This
  // can cause slightly different results with certain special characters.
  Phrase strong_phrase = text::tokenizeSentence(strong_text);
  if (max_len) {
    if (strong_phrase.size() > max_len.value()) {
      strong_phrase.resize(max_len.value());
    }
  }
  return strong_phrase;
}

}  // namespace thirdai::data::cold_start