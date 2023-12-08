#pragma once

#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>
#include <stdexcept>

namespace thirdai::data::cold_start {

using Phrase = std::vector<std::string>;
using PhraseCollection = std::vector<std::vector<std::string>>;

class TextAugmentationBase : public Transformation {
 public:
  TextAugmentationBase(std::vector<std::string> strong_column_names,
                       std::vector<std::string> weak_column_names,
                       std::string label_column_name,
                       std::string output_column_name, uint32_t seed);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  virtual std::vector<std::string> augmentSingleRow(
      const std::string& strong_text, const std::string& weak_text) const = 0;

  ar::ConstArchivePtr toArchive() const final {
    // We never actually serialize cold start transformations because we create
    // them for every call to cold start depending on the strong/weak columns
    // passed in.
    throw std::invalid_argument(
        "Cannot convert cold start transformation to archive.");
  }

 protected:
  std::vector<std::string> _strong_column_names;
  std::vector<std::string> _weak_column_names;
  std::string _label_column_name;
  std::string _output_column_name;
  uint32_t _seed;
};

/**
 * Concatenates each element from the weak phrases with the strong phrase.
 * If strong_sample_num_words is provided, this also independently samples
 * from the strong phrase for every weak phrase.
 */
PhraseCollection mergeStrongWithWeak(
    const PhraseCollection& weak_phrases, const Phrase& strong_phrase,
    std::optional<uint32_t> strong_sample_num_words, uint32_t seed);

/**
 * Randomly deletes elements from each phrase, resulting in new phrases.
 * Repeats the process n_sampled_phrases times for each phrase, resulting in
 * (roughly) n_sampled_phrases * phrases.size() new phrases. Note that if a
 * phrase is not long enough to choose words_per_sampled_phrase words, then it
 * is kept but only represented once in the output (not n_sampled_phrases
 * times)
 */
PhraseCollection sampleFromPhrases(const PhraseCollection& phrases,
                                   uint32_t words_per_sampled_phrase,
                                   uint32_t n_sampled_phrases, uint32_t seed);

/**
 * Returns a single phrase that takes in the concatenated string of strong
 * columns and returns a strong phrase (this will just be a cleaned version of
 * the input string, possibly length restricted).
 */
Phrase getStrongPhrase(const std::string& strong_text_in,
                       std::optional<uint32_t> max_len = std::nullopt);

}  // namespace thirdai::data::cold_start