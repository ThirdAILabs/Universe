#pragma once

#include <cereal/access.hpp>
#include <data/src/transformations/DyadicInterval.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace thirdai::data {

struct FeatureEnhancementConfig {
  bool enhance_names = false;
  bool location_features = false;
  bool organization_features = false;
  bool case_features = false;
  bool numerical_features = false;

  explicit FeatureEnhancementConfig(bool set_all_true)
      : enhance_names(set_all_true),
        location_features(set_all_true),
        organization_features(set_all_true),
        case_features(set_all_true),
        numerical_features(set_all_true) {}

  FeatureEnhancementConfig() = default;
};

class NerDyadicDataProcessor
    : std::enable_shared_from_this<NerDyadicDataProcessor> {
 public:
  explicit NerDyadicDataProcessor(
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
      uint32_t dyadic_num_intervals,
      std::optional<FeatureEnhancementConfig> extra_features_config =
          FeatureEnhancementConfig());

  explicit NerDyadicDataProcessor(const ar::Archive& archive);

  ar::ConstArchivePtr toArchive() const;

  static std::shared_ptr<NerDyadicDataProcessor> make(
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
      uint32_t dyadic_num_intervals);

  std::string processToken(const std::vector<std::string>& tokens,
                           uint32_t index) const;

  std::string generateDyadicWindows(std::vector<std::string> tokens,
                                    uint32_t index) const;

 private:
  NerDyadicDataProcessor() {}

  std::string getExtraFeatures(const std::vector<std::string>& tokens,
                               uint32_t index) const;

  std::vector<dataset::TextTokenizerPtr> _target_word_tokenizers;
  uint32_t _dyadic_num_intervals;

  std::optional<FeatureEnhancementConfig> _extra_features_config;

  std::string _target_prefix = "t_";
  std::string _dyadic_previous_prefix = "pp_";
  std::string _dyadic_next_prefix = "np_";

  dataset::TextTokenizerPtr _sentence_tokenizer =
      std::make_shared<dataset::NaiveSplitTokenizer>(' ');

  std::unordered_set<std::string> location_keywords = {
      "lives", "live", "lived", "located", "moved", "near", "adjacent"};

  std::unordered_set<std::string> organization_keywords = {
      "works", "work", "employed",    "announced",
      "inc",   "corp", "corporation", "incorporation"};

  std::unordered_set<std::string> name_keywords = {"name", "called", "i'm",
                                                   "am"};
};
}  // namespace thirdai::data