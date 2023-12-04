#pragma once

#include "TextAugmentationUtils.h"
#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

using cold_start::Phrase;
using cold_start::PhraseCollection;

struct VariableLengthConfig {
  explicit VariableLengthConfig(
      size_t covering_min_length = 5, size_t covering_max_length = 40,
      std::optional<uint32_t> max_covering_samples = std::nullopt,
      size_t slice_min_length = 5,
      std::optional<size_t> slice_max_length = std::nullopt,
      uint32_t num_slices = 7, bool add_whole_doc = true,
      bool prefilter_punctuation = true, uint32_t strong_sample_num_words = 3,
      float word_removal_probability = 0);

  size_t covering_min_length;
  size_t covering_max_length;
  std::optional<uint32_t> max_covering_samples;
  size_t slice_min_length;
  std::optional<size_t> slice_max_length;
  uint32_t num_slices;
  bool add_whole_doc;
  bool prefilter_punctuation;
  uint32_t strong_sample_num_words;
  float word_removal_probability;
};

class VariableLengthColdStart : public cold_start::TextAugmentationBase {
 public:
  VariableLengthColdStart(
      std::vector<std::string> strong_column_names,
      std::vector<std::string> weak_column_names, std::string label_column_name,
      std::string output_column_name,
      const VariableLengthConfig& config = VariableLengthConfig(),
      uint32_t seed = global_random::nextSeed());

  /**
   * Helper method to perform the augmentation of a single row in the input.
   * Returns the augmented phrases from that input row as strings.
   */
  std::vector<std::string> augmentSingleRow(
      const std::string& strong_text, const std::string& weak_text) const final;

 private:
  /**
   * Returns a set of covering samples and random slices according to the
   * parameters specified at construction time.
   */
  PhraseCollection getWeakPhrases(std::string weak_text) const;

  static void addCoveringPhrases(const Phrase& words, PhraseCollection& phrases,
                                 size_t min_len, size_t max_len,
                                 std::optional<size_t> max_covering_samples,
                                 uint32_t seed);

  static void addRandomSlicePhrases(const Phrase& words,
                                    PhraseCollection& phrases, size_t min_len,
                                    std::optional<size_t> max_len_opt,
                                    uint32_t num_slices, uint32_t seed);

  VariableLengthConfig _config;
};

}  // namespace thirdai::data
