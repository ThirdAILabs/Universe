#include "NerDyadicDataProcessor.h"
#include "utils/utils.h"
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <data/src/transformations/TextTokenizer.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <utils/text/StringManipulation.h>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <regex>

namespace thirdai::data {

std::string getNumericalFeatures(const std::string& input) {
  std::string strippedInput = ner::utils::stripNonDigits(input);

  if (!strippedInput.empty()) {
    // if a token contains both alphabets and numbers, it is probably some uin
    // exluding e, x, and t because they can be present in phonenumbers as well
    if (ner::utils::containsAlphabets(input,
                                      /*excluded_alphas=*/{'e', 'x', 't'}) &&
        input.size() >= 6) {
      return "IS_UIN";
    }

    // ssn is generally in the format : xxx.xx.xxxx or xxx xx xxxx or xxxxxxxxx
    if (strippedInput.size() == 9 &&
        (input.size() == 9 || input.size() == 11)) {
      return "SSN";
    }

    // phone numbers
    if ((strippedInput.size() >= 10 && strippedInput.size() <= 16) ||
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
  }
  return "";
}

std::string NerDyadicDataProcessor::getExtraFeatures(
    const std::vector<std::string>& tokens, uint32_t index,
    const std::vector<std::string>& lower_cased_tokens) const {
  if (!_feature_enhancement_config.has_value()) {
    return "";
  }

  std::string extra_features;

  // clean the tokens to be able to match regex and apply heuristics
  std::string current_token = ner::utils::trimPunctuation(tokens[index]);

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

  if (_feature_enhancement_config->enhance_numerical_features) {
    /*
     * If the current token is a number and has surrounding tokens that are also
     * numbers, they probably form a single entity.
     */
    std::string surrounding_numbers = text::stripWhitespace(
        ner::utils::findContiguousNumbers(lower_cased_tokens, index));

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

  if (ner::utils::containsKeywordInRange(
          lower_cased_tokens, start_long, end,
          _feature_enhancement_config->identification_keywords)) {
    extra_features += "CONTAINS_IDENTIFICATION_KEYWORDS ";
  }

  if (_feature_enhancement_config->enhance_names &&
      ner::utils::containsKeywordInRange(
          lower_cased_tokens, start, end,
          _feature_enhancement_config->name_keywords)) {
    extra_features += "CONTAINS_NAMED_WORDS ";
  }

  if (_feature_enhancement_config->enhance_location_features &&
      ner::utils::containsKeywordInRange(
          lower_cased_tokens, start, end,
          _feature_enhancement_config->location_keywords)) {
    extra_features += "CONTAINS_LOCATION_WORDS ";
  }

  if (_feature_enhancement_config->enhance_organization_features &&
      ner::utils::containsKeywordInRange(
          lower_cased_tokens, start, end,
          _feature_enhancement_config->organization_keywords)) {
    extra_features += "CONTAINS_ORGANIZATION_WORDS ";
  }
  if (_feature_enhancement_config->find_phonenumbers &&
      ner::utils::containsKeywordInRange(
          lower_cased_tokens, start_long, end,
          _feature_enhancement_config->contact_keywords)) {
    extra_features += "CONTAINS_PHONE_WORDS_LONG ";
  }

  return extra_features;
}

NerDyadicDataProcessor::NerDyadicDataProcessor(
    std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
    uint32_t dyadic_num_intervals,
    std::optional<FeatureEnhancementConfig> feature_enhancement_config,
    bool for_inference)
    : _target_word_tokenizers(std::move(target_word_tokenizers)),
      _dyadic_num_intervals(dyadic_num_intervals),
      _feature_enhancement_config(std::move(feature_enhancement_config)),
      _for_inference(for_inference) {}

std::shared_ptr<NerDyadicDataProcessor> NerDyadicDataProcessor::make(
    std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
    uint32_t dyadic_num_intervals,
    std::optional<FeatureEnhancementConfig> feature_enhancement_config,
    bool for_inference) {
  return std::make_shared<NerDyadicDataProcessor>(
      std::move(target_word_tokenizers), dyadic_num_intervals,
      std::move(feature_enhancement_config), for_inference);
}

std::string NerDyadicDataProcessor::processToken(
    const std::vector<std::string>& tokens, uint32_t index,
    const std::vector<std::string>& lower_cased_tokens) const {
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

  // randomly dropping out the target token so that model can learn from the
  // context and does not overfit just on the tokens

  // TODO(@shubh3ai) : make the dropout ratio configurable
  // if number of tokens is 1 or if in inference mode, do not drop the target
  // token
  if (_for_inference || rand() % 2 == 0 || tokens.size() == 1) {
    for (const auto& tokenizer : _target_word_tokenizers) {
      // to change the target token tokenization, change the first argument of
      // toStrings here. example, if you want to remove punct from target token,
      // call the remove punct func and pass the value here
      auto tokens =
          tokenizer->toStrings(ner::utils::trimPunctuation(target_token));
      tokenized_target_token.reserve(tokenized_target_token.size() +
                                     tokens.size());
      tokenized_target_token.insert(tokenized_target_token.end(),
                                    tokens.begin(), tokens.end());
    }
  }

  /*
   * We do not perform deduplication over the tokens returned by the tokenizers.
   * Hence, same tokens can be appended to the string multiple times.
   */
  std::string repr;

  for (const auto& tok : tokenized_target_token) {
    repr += _target_prefix + tok + " ";
  }

  // to use lower cased tokenization for the context or any other
  // modifications, change the first argument to the function here
  repr += generateDyadicWindows(lower_cased_tokens, index);

  if (_feature_enhancement_config.has_value()) {
    repr += " " + getExtraFeatures(tokens, index, lower_cased_tokens);
  }

  if (n_digit > 0) {
    repr += " " + std::to_string(n_punct) + "_PUNCT";
    repr += " " + std::to_string(n_alpha) + "_ALPHA";
    repr += " " + std::to_string(n_digit) + "_DIGIT";
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

  map->set("for_inference", ar::boolean(_for_inference));

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

  _for_inference = archive.getOr<ar::Boolean>("for_inference", true);

  _target_prefix = archive.str("target_prefix");
  _dyadic_previous_prefix = archive.str("dyadic_previous_prefix");
  _dyadic_next_prefix = archive.str("dyadic_next_prefix");
}

}  // namespace thirdai::data
