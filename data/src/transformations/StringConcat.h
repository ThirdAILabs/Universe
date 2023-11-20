#pragma once

#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

/**
 * Concatenates string columns together in the order they are specified.
 */
class StringConcat final : public Transformation {
 public:
  StringConcat(std::vector<std::string> input_column_names,
               std::string output_column_name, std::string seperator = "")
      : _input_column_names(std::move(input_column_names)),
        _output_column_name(std::move(output_column_name)),
        _seperator(std::move(seperator)) {}

  explicit StringConcat(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "string_concat"; }

 private:
  std::vector<std::string> _input_column_names;
  std::string _output_column_name;
  std::string _seperator;
};

}  // namespace thirdai::data