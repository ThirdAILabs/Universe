#pragma once

#include <cereal/access.hpp>
#include " NerDataProcessor.h"
#include <data/src/transformations/DyadicInterval.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <cstdint>
#include <memory>
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

  std::pair<std::vector<std::vector<uint32_t>>,
            std::vector<std::vector<uint32_t>>>
  featurizeTokenTagList(const std::vector<std::string>& tokens,
                        std::vector<uint32_t> tags) {
    std::vector<std::vector<uint32_t>> features(tokens.size());
    std::vector<std::vector<uint32_t>> labels(tags.size());

    for (uint32_t index = 0; index < tokens.size(); index++) {
      std::string featurized_string = processToken(tokens, index);
      features[index] = _sentence_tokenizer->tokenize(featurized_string);
      labels[index] =
          std::vector<uint32_t>({static_cast<uint32_t>(tags[index])});
    }
    return std::make_pair(features, labels);
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
      repr += _target_prefix + tok + " ";
    }

    repr += generateDyadicWindows(tokens, index);
    return repr;
  }

  std::string generateDyadicWindows(std::vector<std::string> tokens,
                                    uint32_t index) {
    // auto print_vector = [](const auto& vec) {
    //   for (const auto& element : vec) {
    //     std::cout << element << " ";
    //   }
    //   std::cout << std::endl;
    // };

    // print_vector(tokens);

    std::vector<std::vector<std::string>> dyadic_windows;
    for (size_t interval_id = 0; interval_id < _dyadic_num_intervals;
         interval_id++) {
      uint32_t interval_size = 1 << interval_id;

      // std::cout << interval_id << std::endl;
      // std::cout << interval_size << std::endl;

      std::vector<std::string> prev_window, next_window;
      prev_window.reserve(interval_size);
      next_window.reserve(interval_size);

      for (size_t lower_index =
               std::max(index - interval_size, static_cast<uint32_t>(0));
           lower_index < index; lower_index++) {
        prev_window.push_back(_dyadic_previous_prefix +
                              std::to_string(interval_id) + "_" +
                              tokens[lower_index]);
      }

      // print_vector(prev_window);

      for (size_t upper_index = std::min(
               index + interval_size, static_cast<uint32_t>(tokens.size() - 1));
           upper_index > index; upper_index--) {
        next_window.push_back(_dyadic_next_prefix +
                              std::to_string(interval_id) + "_" +
                              tokens[upper_index]);
      }
      // print_vector(next_window);

      dyadic_windows.push_back(prev_window);
      dyadic_windows.push_back(next_window);
    }

    std::string repr;
    for (const auto& window : dyadic_windows) {
      for (const auto& tok : window) {
        repr += tok + " ";
      }
    }

    // std::cout << repr << std::endl;
    return repr;
  }

 private:
  SimpleDataProcessor() {}
  friend class cereal::access;

  std::string _tokens_column, _tags_column;
  uint32_t _fhr_dim;
  std::vector<dataset::TextTokenizerPtr> _target_word_tokenizers;
  uint32_t _dyadic_num_intervals;

  std::string _target_prefix = "t_";
  std::string _dyadic_previous_prefix = "pp_";
  std::string _dyadic_next_prefix = "np_";

  dataset::TextTokenizerPtr _sentence_tokenizer =
      std::make_shared<dataset::NaiveSplitTokenizer>(' ');
};
}  // namespace thirdai::bolt