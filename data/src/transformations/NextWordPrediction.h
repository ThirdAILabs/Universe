#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>

namespace thirdai::data {
class NextWordPrediction final : public Transformation {
 public:
  NextWordPrediction(std::string input_column, std::string context_column,
                     std::string target_column);

  explicit NextWordPrediction(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "next_word_prediction"; }

 private:
  std::vector<size_t> computeOffsets(
      const ArrayColumnBasePtr<uint32_t>& texts) const;
  std::string _input_column;
  std::string _context_column;
  std::string _target_column;
};

}  // namespace thirdai::data