#pragma once

#include <cereal/access.hpp>
#include <data/src/transformations/DyadicInterval.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace thirdai::data {
class NerDyadicDataProcessor
    : std::enable_shared_from_this<NerDyadicDataProcessor> {
 public:
  explicit NerDyadicDataProcessor(
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
      uint32_t dyadic_num_intervals);

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

  std::vector<dataset::TextTokenizerPtr> _target_word_tokenizers;
  uint32_t _dyadic_num_intervals;

  std::string _target_prefix = "t_";
  std::string _dyadic_previous_prefix = "pp_";
  std::string _dyadic_next_prefix = "np_";

  dataset::TextTokenizerPtr _sentence_tokenizer =
      std::make_shared<dataset::NaiveSplitTokenizer>(' ');
};
}  // namespace thirdai::data