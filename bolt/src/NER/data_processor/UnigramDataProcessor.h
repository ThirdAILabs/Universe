#pragma once

#include <cereal/access.hpp>
#include " NerDataProcessor.h"
#include <data/src/transformations/DyadicInterval.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace thirdai::bolt {
class SimpleDataProcessor : std::enable_shared_from_this<SimpleDataProcessor>,
                            NerDataProcesser {
 public:
  explicit SimpleDataProcessor(
      std::string tokens_column, std::string tags_column, uint32_t fhr_dim,
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
      uint32_t dyadic_num_intervals)
      : _tokens_column(std::move(tokens_column)),
        _tags_column(std::move(tags_column)),
        _fhr_dim(fhr_dim),
        _target_word_tokenizers(std::move(target_word_tokenizers)),
        _dyadic_num_intervals(dyadic_num_intervals) {}

  static std::shared_ptr<SimpleDataProcessor> make(
      std::string tokens_column, std::string tags_column, uint32_t fhr_dim,
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
      uint32_t dyadic_num_intervals) {
    return std::make_shared<SimpleDataProcessor>(
        tokens_column, tags_column, fhr_dim, target_word_tokenizers,
        dyadic_num_intervals);
  }

  std::string processToken(std::vector<std::string> tokens, uint32_t index) {
    /*
     * 1. Generate Dyadic Intervals for the token
     * 2. For the target word, generate the tokenized word and all.
     * 3. Combine everything into a single string and return it.
     */

    std::string target_token = tokens[index];

    std::vector<std::string> tokenized_target_token;

    for (const auto& tokenizer : _target_word_tokenizers) {
      auto tokens = tokenizer->toStrings(target_token);
      tokenized_target_token.reserve(tokenized_target_token.size() +
                                     tokens.size());
      tokenized_target_token.insert(tokenized_target_token.end(),
                                    tokens.begin(), tokens.end());
    }

    std::string repr;
    for (const auto& tok : tokenized_target_token) {
      repr += tok + " ";
    }

    repr += generateDyadicWindows(tokens, index);
    return repr;
  }

  std::string generateDyadicWindows(std::vector<std::string> tokens,
                                    uint32_t index) {
    std::vector<std::vector<std::string>> dyadic_windows;
    for (size_t interval_id = 0; interval_id < _dyadic_num_intervals;
         interval_id++) {
      uint32_t interval_size = 2 ^ interval_id;

      std::vector<std::string> prev_window, next_window;
      prev_window.reserve(interval_size);
      next_window.reserve(interval_size);

      for (size_t lower_index =
               std::max(index - interval_size, static_cast<uint32_t>(0));
           lower_index < index; lower_index++) {
        prev_window.push_back(dyadic_previous_prefix +
                              std::to_string(interval_id) + "_" +
                              tokens[lower_index]);
      }

      for (size_t upper_index = std::min(index + interval_size,
                                         static_cast<uint32_t>(tokens.size()));
           upper_index > index; upper_index--) {
        prev_window.push_back(dyadic_next_prefix + std::to_string(interval_id) +
                              "_" + tokens[upper_index]);
      }
    }

    std::string repr;
    for (const auto& window : dyadic_windows) {
      for (const auto& tok : window) {
        repr += tok + " ";
      }
    }
    return repr;
  }

 private:
  SimpleDataProcessor() {}
  friend class cereal::access;

  std::string _tokens_column, _tags_column;
  uint32_t _fhr_dim;
  std::vector<dataset::TextTokenizerPtr> _target_word_tokenizers;
  uint32_t _dyadic_num_intervals;

  std::string target_prefix = "t_";
  std::string dyadic_previous_prefix = "pp_";
  std::string dyadic_next_prefix = "np_";
};
}  // namespace thirdai::bolt