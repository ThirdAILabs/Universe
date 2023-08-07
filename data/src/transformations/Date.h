#pragma once

#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class Date final : public Transformation {
 public:
  Date(std::string input_column_name, std::string output_column_name,
       std::string format = "%Y-%m-%d");

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _input_column_name;
  std::string _output_column_name;
  std::string _format;
};

}  // namespace thirdai::data