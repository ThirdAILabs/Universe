#include "NerDyadicDataProcessor.h"
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <data/src/transformations/TextTokenizer.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <utils/text/StringManipulation.h>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <regex>

namespace thirdai::data {

std::string trimPunctuation(const std::string& str) {
  const std::string punctuation = ".,?-!;:";
  size_t start = str.find_first_not_of(punctuation);
  if (start == std::string::npos) {
    return str;
  }
  size_t end = str.find_last_not_of(punctuation);
  return str.substr(start, end - start + 1);
}

std::vector<std::string> cleanAndLowerCase(
    const std::vector<std::string>& tokens) {
  /*
   * Converts the tokens to lower case and trims punctuations.
   */
  std::vector<std::string> lower_tokens(tokens.size());
  std::transform(tokens.begin(), tokens.end(), lower_tokens.begin(),
                 [](const std::string& token) {
                   std::string lower_token;
                   lower_token.reserve(token.size());
                   std::transform(token.begin(), token.end(),
                                  std::back_inserter(lower_token), ::tolower);
                   return lower_token;
                 });
  for (auto& token : lower_tokens) {
    token = trimPunctuation(token);
  }
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

bool is_number_with_punct(const std::string& s) {
  bool has_digit = false;

  for (char c : s) {
    if (std::isdigit(c)) {
      has_digit = true;
    } else if (!std::ispunct(c)) {
      return false;
    }
  }

  return has_digit;
}

bool luhnCheck(const std::string& number) {
  /*
   * Checks whether the number being passed satisifies the luhn's check. This is
   * useful for detecting credit card numbers.
   */
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

std::string find_contiguous_numbers(const std::vector<std::string>& v,
                                    uint32_t index, uint32_t k = 3) {
  /*
   * Returns the surrounding numbers around the target token as a space
   * seperated string. This is useful when we have tokens of the form 1234 5678
   * 9101.
   */
  if (index >= static_cast<uint32_t>(v.size())) {
    return "";
  }

  if (!is_number_with_punct(v[index])) {
    return "";
  }

  int start = index > k ? index - k : 0;
  int end = std::min(static_cast<uint32_t>(v.size()) - 1, index + k);

  std::vector<std::string> left_window, right_window;
  for (int i = index - 1; i >= start; --i) {
    if (is_number_with_punct(v[i])) {
      left_window.push_back(v[i]);
    } else {
      break;
    }
  }

  std::reverse(left_window.begin(), left_window.end());

  for (int i = index + 1; i <= end; ++i) {
    if (is_number_with_punct(v[i])) {
      right_window.push_back(v[i]);
    } else {
      break;
    }
  }
  if (left_window.empty() && right_window.empty()) {
    return "";
  }

  std::string result;
  for (const auto& s : left_window) {
    result += s + " ";
  }
  result += v[index] + " ";
  for (const auto& s : right_window) {
    result += s + " ";
  }

  return result;
}

std::string getNumericalFeatures(const std::string& input) {
  std::string strippedInput = stripNonDigits(input);

  if (!strippedInput.empty()) {
    // useful for credit cards or other account numbers
    if ((luhnCheck(strippedInput) && strippedInput.size() >= 10) ||
        strippedInput.size() > 12) {
      return "IS_ACCOUNT_NUMBER";
    }

    // if a token contains both alphabets and numbers, it is probably some uin
    if (containsAlphabets(input) && input.size() >= 6) {
      return "IS_UIN";
    }

    // ssn is generally in the format : xxx.xx.xxxx or xxx xx xxxx or xxxxxxxxx
    if (strippedInput.size() == 9 &&
        (input.size() == 9 || input.size() == 11)) {
      return "SSN";
    }

    // phone numbers
    if ((strippedInput.size() >= 10 && strippedInput.size() <= 12) ||
        input[0] == '+' || input[0] == '(' || input.back() == ')') {  // NOLINT
      return "MAYBE_PHONE";
    }

    /*zipcode (5 digits) or pincode (6 digits)*/
    /*zipcode with extension*/
    if ((strippedInput.size() <= 6 && strippedInput.size() >= 5) ||
        (strippedInput.size() == 9 && input.size() == 10)) {
      return "MAYBE_ZIP_CODE";
    }

    if (strippedInput.size() == input.size() && strippedInput.size() >= 5) {
      return "IS_NUMBER_OR_UIN";
    }

    /*month or day or year(between 1800 to 2199)*/
    if ((strippedInput.size() <= 2 && std::stoi(strippedInput) <= 31) ||
        (strippedInput.size() == 4 &&
         (std::stoi(strippedInput.substr(0, 2)) <= 21 ||
          std::stoi(strippedInput.substr(0, 2)) >= 18))) {
      return "A_DATE";
    }
  }
  return "";
}

std::string NerDyadicDataProcessor::getExtraFeatures(
    const std::vector<std::string>& tokens, uint32_t index) const {
  if (!_feature_enhancement_config.has_value()) {
    return "";
  }

  std::string extra_features;

  // clean the tokens to be able to match regex and apply heuristics
  std::string current_token = trimPunctuation(tokens[index]);
  auto lower_cased_tokens = cleanAndLowerCase(tokens);

  /*
   * start, end, start_long are indices in the vector that mark the boundary of
   * the context for a token. for a token, we search for certain keywords within
   * this context and add a feature if the keyword is found. ex : tokens = [His,
   * name, is, John], index = 3 since name is in the context of John, we will
   * add feature CONTAINS_NAMED_WORDS to extra_features.
   */
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
    /*
     * If the current token is a number and has surrounding tokens that are also
     * numbers, they probably form a single entity.
     */
    std::string surrounding_numbers = text::stripWhitespace(
        find_contiguous_numbers(lower_cased_tokens, index));
    if (!surrounding_numbers.empty()) {
      auto numerical_features = getNumericalFeatures(surrounding_numbers);
      if (!numerical_features.empty()) {
        extra_features = numerical_features;
      }
    } else {
      auto numerical_features = getNumericalFeatures(current_token);
      if (!numerical_features.empty()) {
        extra_features += numerical_features;
      }
    }

    if (extra_features == "IS_ACCOUNT_NUMBER") {
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
  }

  if (_feature_enhancement_config->enhance_location_features &&
      containsKeywordInRange(lower_cased_tokens, start, end,
                             _feature_enhancement_config->location_keywords)) {
    extra_features += "CONTAINS_LOCATION_WORDS ";
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

  uint32_t n_alpha = 0;
  uint32_t n_digit = 0;
  uint32_t n_punct = 0;
  std::string target_token = tokens[index];
  for (char& c : target_token) {
    if (std::isdigit(c)) {
      c = '#';
      n_digit++;
    } else if (std::isalpha(c)) {
      n_alpha++;
    } else if (std::ispunct(c)) {
      n_punct++;
    }
  }

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

  repr += " " + std::to_string(n_alpha) + "_ALPHA";
  repr += " " + std::to_string(n_digit) + "_DIGIT";
  repr += " " + std::to_string(n_punct) + "_PUNCT";

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

    for (size_t lower_index = interval_size > index ? 0 : index - interval_size;
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

  map->set("target_prefix", ar::str(_target_prefix));
  map->set("dyadic_previous_prefix", ar::str(_dyadic_previous_prefix));
  map->set("dyadic_next_prefix", ar::str(_dyadic_next_prefix));

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

  _target_prefix = archive.str("target_prefix");
  _dyadic_previous_prefix = archive.str("dyadic_previous_prefix");
  _dyadic_next_prefix = archive.str("dyadic_next_prefix");
}

}  // namespace thirdai::data
