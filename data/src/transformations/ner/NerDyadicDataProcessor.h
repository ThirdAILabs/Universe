#pragma once

#include <cereal/access.hpp>
#include <archive/src/Archive.h>
#include <data/src/transformations/DyadicInterval.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <cstdint>
#include <memory>
#include <regex>
#include <string>
#include <unordered_set>
#include <vector>

namespace thirdai::data {

struct FeatureEnhancementConfig {
  bool enhance_names = true;
  bool enhance_location_features = true;
  bool enhance_organization_features = true;
  bool enhance_case_features = true;
  bool enhance_numerical_features = true;
  bool find_emails = true;
  bool find_phonenumbers = true;

  std::unordered_set<std::string> location_keywords = {
      "lives", "live", "lived", "located", "moved", "near", "adjacent"};

  std::unordered_set<std::string> organization_keywords = {
      "works", "work", "employed",    "announced",
      "inc",   "corp", "corporation", "incorporation"};

  std::unordered_set<std::string> name_keywords = {"name", "called", "i'm",
                                                   "am", "named"};

  std::unordered_set<std::string> contact_keywords = {
      "call", "contact", "dial", "mobile", "phone", "cellphone", "cell"};

  std::unordered_set<std::string> identification_keywords = {
      "id",     "identity", "identification", "license",
      "number", "code",     "identifier",     "ssn"};

  std::regex email_regex =
      std::regex(R"((^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z,.]{2,}$))");

  std::regex month_regex = std::regex(
      R"((^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)))");

  std::regex date_regex = std::regex(
      R"((\d{1,2}[-/.:|]\d{1,2}[-/.:|]\d{2,4})|(\d{4}[-/.:|]\d{2}[-/.:|]\d{2}))");

  FeatureEnhancementConfig() = default;

  FeatureEnhancementConfig(bool enhance_names, bool enhance_location_features,
                           bool enhance_organization_features,
                           bool enhance_case_features,
                           bool enhance_numerical_features, bool find_emails,
                           bool find_phonenumbers)
      : enhance_names(enhance_names),
        enhance_location_features(enhance_location_features),
        enhance_organization_features(enhance_organization_features),
        enhance_case_features(enhance_case_features),
        enhance_numerical_features(enhance_numerical_features),
        find_emails(find_emails),
        find_phonenumbers(find_phonenumbers) {}

  explicit FeatureEnhancementConfig(const ar::Archive& archive) {
    enhance_names = archive.getOr<bool>("enhance_names", true);
    enhance_location_features =
        archive.getOr<bool>("enhance_location_features", true);
    enhance_organization_features =
        archive.getOr<bool>("enhance_organization_features", true);
    enhance_case_features = archive.getOr<bool>("enhance_case_features", true);
    enhance_numerical_features =
        archive.getOr<bool>("enhance_numerical_features", true);
    find_emails = archive.getOr<bool>("find_emails", true);
    find_phonenumbers = archive.getOr<bool>("find_phonenumbers", true);
  }

  ar::ConstArchivePtr toArchive() const {
    auto map = ar::Map::make();
    map->set("enhance_names", ar::boolean(enhance_names));
    map->set("enhance_location_features",
             ar::boolean(enhance_location_features));
    map->set("enhance_organization_features",
             ar::boolean(enhance_organization_features));
    map->set("enhance_case_features", ar::boolean(enhance_case_features));
    map->set("enhance_numerical_features",
             ar::boolean(enhance_numerical_features));
    map->set("find_emails", ar::boolean(find_emails));
    map->set("find_phonenumbers", ar::boolean(find_phonenumbers));
    return map;
  }
};

class NerDyadicDataProcessor
    : std::enable_shared_from_this<NerDyadicDataProcessor> {
 public:
  explicit NerDyadicDataProcessor(
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
      uint32_t dyadic_num_intervals,
      std::optional<FeatureEnhancementConfig> feature_enhancement_config,
      bool for_inference);

  explicit NerDyadicDataProcessor(const ar::Archive& archive);

  ar::ConstArchivePtr toArchive() const;

  static std::shared_ptr<NerDyadicDataProcessor> make(
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
      uint32_t dyadic_num_intervals,
      std::optional<FeatureEnhancementConfig> feature_enhancement_config,
      bool for_inference);

  std::string processToken(
      const std::vector<std::string>& tokens, uint32_t index,
      const std::vector<std::string>& lower_cased_tokens) const;

  std::string generateDyadicWindows(std::vector<std::string> tokens,
                                    uint32_t index) const;

  const auto& targetWordTokenizers() const { return _target_word_tokenizers; }

  uint32_t nDyadicIntervals() const { return _dyadic_num_intervals; }

  const auto& featureConfig() const { return _feature_enhancement_config; }

 private:
  NerDyadicDataProcessor() {}

  std::string getExtraFeatures(
      const std::vector<std::string>& tokens, uint32_t index,
      const std::vector<std::string>& lower_cased_tokens) const;

  std::vector<dataset::TextTokenizerPtr> _target_word_tokenizers;
  uint32_t _dyadic_num_intervals;
  std::optional<FeatureEnhancementConfig> _feature_enhancement_config;
  bool _for_inference;

  std::string _target_prefix = "t_";
  std::string _dyadic_previous_prefix = "p_";
  std::string _dyadic_next_prefix = "p_";

  std::vector<dataset::TextTokenizerPtr> _context_tokenizers = {
      dataset::CharKGramTokenizer::make(4),
      dataset::NaiveSplitTokenizer::make()};
};

}  // namespace thirdai::data