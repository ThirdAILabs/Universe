#include "NerDyadicDataProcessor.h"
#include "utils.h"
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <data/src/transformations/TextTokenizer.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <regex>

namespace thirdai::data {

std::string NerDyadicDataProcessor::getExtraFeatures(
    const std::vector<std::string>& tokens, uint32_t index) const {
  if (!_feature_enhancement_config.has_value()) {
    return "";
  }

  std::string extra_features;

  std::string current_token = trimPunctuation(tokens[index]);
  auto lower_cased_tokens = toLowerCaseTokens(tokens);

  size_t start = (index > 1) ? (index - 2) : 0;
  size_t end = std::min(tokens.size(), static_cast<size_t>(index + 3));
  size_t start_long =
      (index > 5) ? (index - 6)
                  : 0;  // we need more context for phone numbers or uins

  if (std::regex_match(lower_cased_tokens[index],
                       _feature_enhancement_config->month_regex) ||
      std::regex_match(lower_cased_tokens[index],
                       _feature_enhancement_config->date_regex)) {
    extra_features += "A_VALID_DATE ";
    return extra_features;
  }

  if (_feature_enhancement_config->find_emails) {
    if (std::regex_match(lower_cased_tokens[index],
                         _feature_enhancement_config->email_regex)) {
      extra_features += "IS_VALID_EMAIL ";
      return extra_features;
    }
  }

  if (_feature_enhancement_config->enhance_numerical_features) {
    std::string surrounding_numbers =
        find_contiguous_numbers(lower_cased_tokens, index);
    if (!surrounding_numbers.empty()) {
      extra_features = getNumericalFeatures(surrounding_numbers);
    } else {
      auto numerical_features = getNumericalFeatures(current_token);
      if (!numerical_features.empty()) {
        extra_features += numerical_features;
      }
    }

    if (extra_features == "IS_ACCOUNT_NUMBER" || extra_features == "IS_PHONE") {
      return extra_features;
    }
    extra_features += " ";
  }

  if (_feature_enhancement_config->enhance_case_features) {
    if (index >= 1 &&
        std::isupper(static_cast<unsigned char>(current_token[0]))) {
      extra_features += "IS_CAPS_LOCK ";
    }

    if (index >= 1) {
      if (std::islower(static_cast<unsigned char>(tokens[index - 1][0]))) {
        extra_features += "PREVIOUS_LOWER ";
      } else {
        extra_features += "PREVIOUS_UPPER ";
      }
    }

    if (index < tokens.size() - 1) {
      if (std::islower(static_cast<unsigned char>(tokens[index + 1][0]))) {
        extra_features += "NEXT_LOWER ";
      } else {
        extra_features += "NEXT_UPPER ";
      }
    }
  }

  if (containsKeywordInRange(
          lower_cased_tokens, start_long, end,
          _feature_enhancement_config->identification_keywords)) {
    extra_features += "CONTAINS_IDENTIFICATION_KEYWORDS ";
  }

  if (_feature_enhancement_config->enhance_names &&
      containsKeywordInRange(lower_cased_tokens, start, end,
                             _feature_enhancement_config->name_keywords)) {
    extra_features += "CONTAINS_NAMED_WORDS ";
    // return extra_features;
  }

  if (_feature_enhancement_config->enhance_location_features &&
      containsKeywordInRange(lower_cased_tokens, start, end,
                             _feature_enhancement_config->location_keywords)) {
    extra_features += "CONTAINS_LOCATION_WORDS ";
    // return extra_features;
  }

  if (_feature_enhancement_config->enhance_organization_features &&
      containsKeywordInRange(
          lower_cased_tokens, start, end,
          _feature_enhancement_config->organization_keywords)) {
    extra_features += "CONTAINS_ORGANIZATION_WORDS ";
  }
  if (_feature_enhancement_config->find_phonenumbers &&
      containsKeywordInRange(lower_cased_tokens, start_long, end,
                             _feature_enhancement_config->contact_keywords)) {
    extra_features += "CONTAINS_PHONE_WORDS_LONG ";
  }

  return extra_features;
}

NerDyadicDataProcessor::NerDyadicDataProcessor(
    std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
    uint32_t dyadic_num_intervals,
    std::optional<FeatureEnhancementConfig> feature_enhancement_config)
    : _target_word_tokenizers(std::move(target_word_tokenizers)),
      _dyadic_num_intervals(dyadic_num_intervals),
      _feature_enhancement_config(std::move(feature_enhancement_config)) {}

std::shared_ptr<NerDyadicDataProcessor> NerDyadicDataProcessor::make(
    std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
    uint32_t dyadic_num_intervals,
    std::optional<FeatureEnhancementConfig> feature_enhancement_config) {
  return std::make_shared<NerDyadicDataProcessor>(
      std::move(target_word_tokenizers), dyadic_num_intervals,
      std::move(feature_enhancement_config));
}

std::string NerDyadicDataProcessor::processToken(
    const std::vector<std::string>& tokens, uint32_t index) const {
  /*
   * Returns a featurized string for the target token and it's context in the
   * sentence.
   * 1. Generate Dyadic Intervals for the token
   * 2. For the target word, generate the tokenized word and all.
   * 3. Combine everything into a single string and return it.
   */

  const std::string& target_token = tokens[index];

  std::vector<std::string> tokenized_target_token;

  for (const auto& tokenizer : _target_word_tokenizers) {
    auto tokens = tokenizer->toStrings(target_token);
    tokenized_target_token.reserve(tokenized_target_token.size() +
                                   tokens.size());
    tokenized_target_token.insert(tokenized_target_token.end(), tokens.begin(),
                                  tokens.end());
  }

  /*
   * We do not perform deduplication over the tokens returned by the tokenizers.
   * Hence, same tokens can be appended to the string multiple times.
   */
  std::string repr;
  for (const auto& tok : tokenized_target_token) {
    repr += _target_prefix + tok + " ";
  }

  repr += generateDyadicWindows(tokens, index);

  if (_feature_enhancement_config.has_value()) {
    repr += " " + getExtraFeatures(tokens, index);
  }

  return repr;
}

std::string NerDyadicDataProcessor::generateDyadicWindows(
    std::vector<std::string> tokens, uint32_t index) const {
  std::vector<std::vector<std::string>> dyadic_windows;
  for (size_t interval_id = 0; interval_id < _dyadic_num_intervals;
       interval_id++) {
    uint32_t interval_size = 1 << interval_id;

    std::vector<std::string> prev_window, next_window;
    prev_window.reserve(interval_size);
    next_window.reserve(interval_size);

    for (size_t lower_index = std::max(index - interval_size, 0U);
         lower_index < index; lower_index++) {
      prev_window.push_back(_dyadic_previous_prefix +
                            std::to_string(interval_id) + "_" +
                            tokens[lower_index]);
    }

    for (size_t upper_index = std::min(
             index + interval_size, static_cast<uint32_t>(tokens.size() - 1));
         upper_index > index; upper_index--) {
      next_window.push_back(_dyadic_next_prefix + std::to_string(interval_id) +
                            "_" + tokens[upper_index]);
    }

    dyadic_windows.push_back(prev_window);
    dyadic_windows.push_back(next_window);
  }

  std::string repr;
  for (const auto& window : dyadic_windows) {
    for (const auto& tok : window) {
      repr += tok + " ";
    }
  }
  return repr;
}

ar::ConstArchivePtr NerDyadicDataProcessor::toArchive() const {
  auto map = ar::Map::make();
  auto tokenizers = ar::List::make();
  for (const auto& t : _target_word_tokenizers) {
    tokenizers->append(t->toArchive());
  }

  map->set("target_word_tokenizers", tokenizers);
  map->set("dyadic_num_intervals", ar::u64(_dyadic_num_intervals));

  if (_feature_enhancement_config.has_value()) {
    map->set("feature_enhancement_config",
             _feature_enhancement_config->toArchive());
  }

  return map;
}

NerDyadicDataProcessor::NerDyadicDataProcessor(const ar::Archive& archive) {
  for (const auto& t : archive.get("target_word_tokenizers")->list()) {
    _target_word_tokenizers.push_back(dataset::TextTokenizer::fromArchive(*t));
  }
  _dyadic_num_intervals = archive.u64("dyadic_num_intervals");

  if (archive.contains("feature_enhancement_config")) {
    _feature_enhancement_config =
        FeatureEnhancementConfig(*archive.get("feature_enhancement_config"));
  } else {
    _feature_enhancement_config = std::nullopt;
  }
}
}  // namespace thirdai::data
