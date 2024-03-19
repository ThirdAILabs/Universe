#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/optional.hpp>
#include "TextAugmentationUtils.h"
#include <archive/src/Archive.h>
#include <data/src/transformations/Transformation.h>
#include <memory>
#include <random>
#include <sstream>

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
      std::optional<uint32_t> strong_to_weak_ratio = std::nullopt,
      float stopword_removal_probability = 0,
      float stopword_insertion_probability = 0,
      float word_removal_probability = 0,
      float word_perturbation_probability = 0,
      size_t chars_replace_with_space = 0, size_t chars_deleted = 0,
      size_t chars_duplicated = 0, size_t chars_replace_with_adjacents = 0);

  size_t covering_min_length;
  size_t covering_max_length;
  std::optional<uint32_t> max_covering_samples;
  size_t slice_min_length;
  std::optional<size_t> slice_max_length;
  uint32_t num_slices;
  bool add_whole_doc;
  bool prefilter_punctuation;
  uint32_t strong_sample_num_words;
  std::optional<uint32_t> strong_to_weak_ratio;
  float stopword_removal_probability;
  float stopword_insertion_probability;
  float word_removal_probability;
  float word_perturbation_probability;
  size_t chars_replace_with_space;
  size_t chars_deleted;
  size_t chars_duplicated;
  size_t chars_replace_with_adjacents;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(covering_min_length, covering_max_length, max_covering_samples,
            slice_min_length, slice_max_length, num_slices, add_whole_doc,
            prefilter_punctuation, strong_sample_num_words,
            strong_to_weak_ratio, stopword_removal_probability,
            stopword_insertion_probability, word_removal_probability,
            word_perturbation_probability, chars_replace_with_space,
            chars_deleted, chars_duplicated, chars_replace_with_adjacents);
  }

  void save_stream(std::ostream& output_stream) const {
    cereal::BinaryOutputArchive oarchive(output_stream);
    oarchive(*this);
  }

  void overrideStrongSampleNumWords(uint32_t override_value) {
    strong_sample_num_words = override_value;
  }

  static std::shared_ptr<VariableLengthConfig> load_stream(
      std::istream& input_stream) {
    cereal::BinaryInputArchive iarchive(input_stream);
    auto config = std::make_shared<VariableLengthConfig>();
    iarchive(*config);
    return config;
  }
  std::string to_string() const {
    std::stringstream ss;
    ss << "VariableLengthConfig(";
    ss << VARIABLE_TO_STRING(covering_min_length, ", ");
    ss << VARIABLE_TO_STRING(covering_max_length, ", ");
    ss << VARIABLE_TO_STRING(max_covering_samples, ", ");
    ss << VARIABLE_TO_STRING(slice_min_length, ", ");
    ss << VARIABLE_TO_STRING(slice_max_length, ", ");
    ss << VARIABLE_TO_STRING(num_slices, ", ");
    ss << VARIABLE_TO_STRING(add_whole_doc, ", ");
    ss << VARIABLE_TO_STRING(prefilter_punctuation, ", ");
    ss << VARIABLE_TO_STRING(strong_sample_num_words, ", ");
    ss << VARIABLE_TO_STRING(strong_to_weak_ratio, ", ");
    ss << VARIABLE_TO_STRING(stopword_removal_probability, ", ");
    ss << VARIABLE_TO_STRING(stopword_insertion_probability, ", ");
    ss << VARIABLE_TO_STRING(word_removal_probability, ", ");
    ss << VARIABLE_TO_STRING(word_perturbation_probability, ", ");
    ss << VARIABLE_TO_STRING(chars_replace_with_space, ", ");
    ss << VARIABLE_TO_STRING(chars_deleted, ", ");
    ss << VARIABLE_TO_STRING(chars_duplicated, ", ");
    ss << VARIABLE_TO_STRING(chars_replace_with_adjacents, ")");
    return ss.str();
  }
};

class VariableLengthColdStart final : public cold_start::TextAugmentationBase {
 public:
  VariableLengthColdStart(
      std::vector<std::string> strong_column_names,
      std::vector<std::string> weak_column_names,
      std::string output_column_name,
      const VariableLengthConfig& config = VariableLengthConfig(),
      uint32_t seed = global_random::nextSeed());

  /**
   * Helper method to perform the augmentation of a single row in the input.
   * Returns the augmented phrases from that input row as strings.
   */
  std::vector<std::string> augmentSingleRow(const std::string& strong_text,
                                            const std::string& weak_text,
                                            uint32_t row_id_salt) const final;

 private:
  Phrase convertTextToPhrase(std::string string) const;

  /**
   * Returns a set of covering samples and random slices according to the
   * parameters specified at construction time.
   */
  PhraseCollection getWeakPhrases(std::string weak_text,
                                  std::mt19937& rng) const;

  static void addCoveringPhrases(const Phrase& words, PhraseCollection& phrases,
                                 size_t min_len, size_t max_len,
                                 std::optional<size_t> max_covering_samples,
                                 std::mt19937& rng);

  static void addRandomSlicePhrases(const Phrase& words,
                                    PhraseCollection& phrases, size_t min_len,
                                    std::optional<size_t> max_len_opt,
                                    uint32_t num_slices, std::mt19937& rng);

  static std::string convertPhraseToText(const std::vector<std::string>& phrase,
                                         float stopword_removal_probability,
                                         float stopword_insertion_probability,
                                         float word_removal_probability,
                                         float word_perturbation_probability,
                                         std::mt19937& rng);

  VariableLengthConfig _config;
};

}  // namespace thirdai::data
