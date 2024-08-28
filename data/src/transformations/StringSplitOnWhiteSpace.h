#pragma once

#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <string>
#include <vector>

namespace thirdai::data {

/**
 * Splits strings in a column based on whitespace.
 */
class StringSplitOnWhiteSpace final : public Transformation {
 public:
  StringSplitOnWhiteSpace(std::string input_column_name,
                          std::string output_column_name)
      : _input_column_name(std::move(input_column_name)),
        _output_column_name(std::move(output_column_name)) {}

  explicit StringSplitOnWhiteSpace(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "string_split_whitespace"; }

 private:
  std::string _input_column_name;
  std::string _output_column_name;
};

}  // namespace thirdai::data
