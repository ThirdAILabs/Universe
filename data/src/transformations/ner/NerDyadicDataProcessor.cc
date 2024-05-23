#include "NerDyadicDataProcessor.h"
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
#include <unordered_set>

namespace thirdai::data {

std::vector<std::string> toLowerCaseTokens(
    const std::vector<std::string>& tokens) {
  std::vector<std::string> lower_tokens(tokens.size());
  std::transform(tokens.begin(), tokens.end(), lower_tokens.begin(),
                 [](const std::string& token) {
                   std::string lower_token;
                   lower_token.reserve(token.size());
                   std::transform(token.begin(), token.end(),
                                  std::back_inserter(lower_token), ::tolower);
                   return lower_token;
                 });
  return lower_tokens;
}

bool containsKeywordInRange(const std::vector<std::string>& tokens,
                            size_t start, size_t end,
                            const std::unordered_set<std::string>& keywords) {
  return std::any_of(tokens.begin() + start, tokens.begin() + end,
                     [&keywords](const std::string& token) {
                       return keywords.find(token) != keywords.end();
                     });
}

std::string stripNonDigits(const std::string& input) {
  std::string digits;
  for (char ch : input) {
    if (std::isdigit(ch)) {
      digits += ch;
    }
  }
  return digits;
}

bool containsAlphabets(const std::string& input) {
  return std::any_of(input.begin(), input.end(), ::isalpha);
}

bool luhnCheck(const std::string& number) {
  int sum = 0;
  bool alternate = false;
  for (int i = number.size() - 1; i >= 0; --i) {
    int n = number[i] - '0';
    if (alternate) {
      n *= 2;
      if (n > 9) {
        n -= 9;
      }
    }
    sum += n;
    alternate = !alternate;
  }
  return (sum % 10 == 0);
}

std::string getNumericalFeatures(const std::string& input) {
  std::string strippedInput = stripNonDigits(input);

  if (!strippedInput.empty()) {
    if (luhnCheck(strippedInput)) {
      return "IS_ACCOUNT_NUMBER ";
    }

    if (containsAlphabets(input) && input.size() >= 6) {
      return "MAYBE_UIN";
    }

    if ((strippedInput.size() >= 9 && strippedInput.size() <= 12) ||
        input[0] == '+' || input[0] == '(' || input.back() == ')') {  // NOLINT
      return "MAYBE_PHONE ";
    }

    if (strippedInput.size() == input.size() && strippedInput.size() >= 5) {
      return "IS_NUMBER_OR_UIN ";
    }

    if ((strippedInput.size() <= 2 && std::stoi(strippedInput) <= 31) ||
        strippedInput.size() == 4 || strippedInput.size() == 8) {
      return "A_DATE ";
    }

    if (strippedInput.size() <= 6 && strippedInput.size() >= 5) {
      return "MAYBE_ZIP_CODE";
    }
  }
  return "";
}

bool isValidEmail(const std::string& email) {
  const std::regex email_regex(
      R"((^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z,.]{2,}$))");
  return std::regex_match(email, email_regex);
}

bool isValidDate(const std::string& token) {
  // Check if the token matches the regex pattern
  const std::regex month(
      R"((^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)))");
  return std::regex_match(token, month);
}

std::string trimPunctuation(const std::string& str) {
  const std::string punctuation = ".,?-!;:";
  size_t start = str.find_first_not_of(punctuation);
  if (start == std::string::npos) {
    return str;
  }
  size_t end = str.find_last_not_of(punctuation);
  return str.substr(start, end - start + 1);
}

std::string NerDyadicDataProcessor::getExtraFeatures(
    const std::vector<std::string>& tokens, uint32_t index) const {
  if (!_feature_enhancement_config.has_value()) {
    return "";
  }

  std::string extra_features;

  std::string current_token = trimPunctuation(tokens[index]);
  auto lower_cased_tokens = toLowerCaseTokens(tokens);

  if (isValidDate(lower_cased_tokens[index])) {
    extra_features += "A_VALID_DATE ";
    return extra_features;
  }

  if (_feature_enhancement_config->find_emails) {
    if (isValidEmail(lower_cased_tokens[index])) {
      extra_features += "IS_VALID_EMAIL ";
      return extra_features;
    }
  }

  if (_feature_enhancement_config->enhance_numerical_features) {
    auto numerical_features = getNumericalFeatures(current_token);
    if (!numerical_features.empty()) {
      extra_features += numerical_features;
      return extra_features;
    }
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

  size_t start = (index > 1) ? (index - 2) : 0;
  size_t end = std::min(tokens.size(), static_cast<size_t>(index + 3));

  if (_feature_enhancement_config->enhance_names &&
      containsKeywordInRange(lower_cased_tokens, start, end,
                             _feature_enhancement_config->name_keywords)) {
    extra_features += "CONTAINS_NAMED_WORDS ";
    return extra_features;
  }

  if (_feature_enhancement_config->enhance_location_features &&
      containsKeywordInRange(lower_cased_tokens, start, end,
                             _feature_enhancement_config->location_keywords)) {
    extra_features += "CONTAINS_LOCATION_WORDS ";
    return extra_features;
  }

  if (_feature_enhancement_config->enhance_organization_features &&
      containsKeywordInRange(
          lower_cased_tokens, start, end,
          _feature_enhancement_config->organization_keywords)) {
    extra_features += "CONTAINS_ORGANIZATION_WORDS ";
  }

  size_t start_long =
      (index > 5) ? (index - 6) : 0;  // we need more context for phone numbers
  if (_feature_enhancement_config->find_phonenumbers &&
      containsKeywordInRange(lower_cased_tokens, start_long, end,
                             _feature_enhancement_config->contact_keywords)) {
    extra_features += "CONTAINS_PHONE_WORDS_LONG ";
    return extra_features;
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
