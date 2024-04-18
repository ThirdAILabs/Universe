#pragma once

#include <auto_ml/src/featurization/DataTypes.h>
#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>
#include <optional>

namespace thirdai::data {

class NextWordPredictionDualTokenizer final : public Transformation {
 public:
  NextWordPredictionDualTokenizer(std::string input_column,
                                  std::string context_column,
                                  std::string target_column,
                                  std::string input_tokenizer,
                                  std::string output_tokenizer);

  explicit NextWordPredictionDualTokenizer(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "next_word_prediction_dual_tokenizer"; }

 private:
  static std::vector<size_t> computeOffsets(
      const std::vector<std::vector<uint32_t>>& target_offsets);
  std::string _input_column;
  std::string _context_column;
  std::string _target_column;
  dataset::TextTokenizerPtr _input_tokenizer;
  dataset::TextTokenizerPtr _output_tokenizer;
  std::string _input_tokenizer_type;
  std::string _output_tokenizer_type;
};

}  // namespace thirdai::data