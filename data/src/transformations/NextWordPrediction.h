#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

struct NextWordPredictionConfig{
  
  NextWordPredictionConfig(std::string input_column, uint32_t vocab_size);

  std::string input_column;
  uint32_t vocab_size;
};

class NextWordPrediction final : public Transformation {
 public:
  NextWordPrediction(std::string input_column, std::string context_column,
                     std::string target_column, std::optional<std::string> value_column = std::nullopt);

  explicit NextWordPrediction(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "next_word_prediction"; }

 private:
  static std::vector<size_t> computeOffsets(
      const ArrayColumnBasePtr<uint32_t>& texts) ;

  std::string _input_column;
  std::string _context_column;
  std::string _target_column;
  std::optional<std::string> _value_column;
};

}  // namespace thirdai::data