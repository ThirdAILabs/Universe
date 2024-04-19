#pragma once

#include <archive/src/Archive.h>
#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class ConcatTokens final : public Transformation {
 public:
  ConcatTokens(std::vector<std::string> input_cols, std::string output_col)
      : _input_cols(std::move(input_cols)),
        _output_col(std::move(output_col)) {}

  explicit ConcatTokens(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "concat_tokens"; }

 private:
  std::vector<std::string> _input_cols;
  std::string _output_col;
};

}  // namespace thirdai::data