#include "TextAugmentationUtils.h"
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/StringConcat.h>
#include <utils/text/StringManipulation.h>
#include <random>

namespace thirdai::data::cold_start {

TextAugmentationBase::TextAugmentationBase(
    std::vector<std::string> strong_column_names,
    std::vector<std::string> weak_column_names, std::string output_column_name,
    uint32_t seed)
    : _strong_column_names(std::move(strong_column_names)),
      _weak_column_names(std::move(weak_column_names)),

      _output_column_name(std::move(output_column_name)),
      _seed(seed) {}

ColumnMap TextAugmentationBase::apply(ColumnMap columns, State& state) const {
  (void)state;

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

  std::vector<std::string> augmented_data;
  std::vector<size_t> perm;

  std::exception_ptr exception = nullptr;

#pragma omp parallel for default(none) \
    shared(strong_column, weak_column, augmented_data, perm, exception)
  for (uint64_t row_id = 0; row_id < strong_column->numRows(); row_id++) {
    try {
      std::string strong_text = strong_column->value(row_id);
      std::string weak_text = weak_column->value(row_id);

      std::vector<std::string> augmented_samples =
          augmentSingleRow(strong_text, weak_text, /* row_id_salt= */ row_id);

#pragma omp critical
      {
        for (auto& sample : augmented_samples) {
          if (!sample.empty()) {
            augmented_data.emplace_back(std::move(sample));
            perm.push_back(row_id);
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

  columns.dropColumn("strong_text");
  columns.dropColumn("weak_text");
  for (const auto& col : _strong_column_names) {
    // To handle when a column is in both the strong and weak columns, or if an
    // input column is called "strong_text" or "weak_text".
    if (columns.containsColumn(col)) {
      columns.dropColumn(col);
    }
  }
  for (const auto& col : _weak_column_names) {
    // To handle when a column is in both the strong and weak columns, or if an
    // input column is called "strong_text" or "weak_text".
    if (columns.containsColumn(col)) {
      columns.dropColumn(col);
    }
  }

  ColumnMap new_columns = columns.permute(perm);
  new_columns.setColumn(_output_column_name, ValueColumn<std::string>::make(
                                                 std::move(augmented_data)));

  return new_columns;
}

PhraseCollection mergeStrongWithWeak(
    const PhraseCollection& weak_phrases, const Phrase& strong_phrase,
    std::optional<uint32_t> strong_sample_num_words,
    std::optional<uint32_t> strong_to_weak_ratio, std::mt19937& rng) {
  if (weak_phrases.empty()) {
    return {strong_phrase};
  }

  PhraseCollection downsampled_strong_phrases;

  if (strong_sample_num_words) {
    // If we have to sample from the strong phrase, we create N independently
    // sampled sub-strings, where N is the number of weak phrases that we have.
    uint32_t n_sampled_phrases =
        strong_to_weak_ratio
            ? weak_phrases.size() * strong_to_weak_ratio.value()
            : weak_phrases.size();
    downsampled_strong_phrases = sampleFromPhrases(
        /* phrases= */ {strong_phrase},
        /* words_per_sampled_phrase= */ strong_sample_num_words.value(),
        /* n_sampled_phrases= */ n_sampled_phrases, rng);
  }
  PhraseCollection output_phrases;

  if (strong_to_weak_ratio) {
    output_phrases.reserve(1 + weak_phrases.size() +
                           downsampled_strong_phrases.size());

    for (const auto& weak_phrase : weak_phrases) {
      output_phrases.emplace_back(weak_phrase);
    }
    for (const auto& str_phrase : downsampled_strong_phrases) {
      output_phrases.emplace_back(str_phrase);
    }

    output_phrases.emplace_back(strong_phrase);
  } else {
    // returns to the older logic of concatenating weak and strong phrases
    output_phrases.reserve(weak_phrases.size());
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
      output_phrases.emplace_back(std::move(concat_phrase));
    }
  }
  return output_phrases;
}
PhraseCollection sampleFromPhrases(const PhraseCollection& phrases,
                                   uint32_t words_per_sampled_phrase,
                                   uint32_t n_sampled_phrases,
                                   std::mt19937& rng) {
  // Only iterate over the original phrases, as we append new ones to the end.
  if (n_sampled_phrases == 0) {
    throw std::invalid_argument(
        "Invalid number of sampling repetitions: "
        "must be greater than 0.");
  }
  PhraseCollection output_phrases;
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

}  // namespace thirdai::data::cold_start