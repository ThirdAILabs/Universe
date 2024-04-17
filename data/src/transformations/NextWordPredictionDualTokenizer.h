#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <optional>

namespace thirdai::data {

// struct NWPDualTokenizerConfig {
//   NWPDualTokenizerConfig(std::optional<std::string> input_tokenizer,
//                          std::optional<std::string> output_tokenizer,
//                          std::optional<std::string&> input_tokenizer_vocab,
//                          bool input_lowercase,
//                          std::optional<std::string&> output_tokenizer_vocab,
//                          bool output_lowercase, ) {
//     if (input_tokenizer_vocab.has_value()) {
//       _input_tokenizer = std::make_shared<dataset::WordpieceTokenizer>(
//           input_tokenizer_vocab, input_lowercase)
//     }
//     else if (input_tokenizer.has_value()){
//       if (input_tokenizer == "char-")
//     }
//   }
//   data::TextTokenizer _input_tokenizer;
//   data::TextTokenizer _output_tokenizer;
// }

class NextWordPredictionDualTokenizer final : public Transformation {
 public:
  NextWordPredictionDualTokenizer(std::string input_column,
                                  std::string context_column,
                                  std::string target_column);

  explicit NextWordPredictionDualTokenizer(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "next_word_prediction_dual_tokenizers"; }

 private:
  std::vector<size_t> computeOffsets(
      const ArrayColumnBasePtr<uint32_t>& texts) const;
  std::string _input_column;
  std::string _context_column;
  std::string _target_column;
  data::TextTokenizer _input_tokenizer;
  data::TextTokenizer _output_tokenizer;
};

}  // namespace thirdai::data