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

  explicit StringConcat(const proto::data::StringConcat& string_concat);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  proto::data::Transformation* toProto() const final;

 private:
  std::vector<std::string> _input_column_names;
  std::string _output_column_name;
  std::string _seperator;
};

}  // namespace thirdai::data