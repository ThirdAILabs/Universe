#pragma once

#include <data/src/transformations/Transformation.h>
#include <string>
#include <vector>

namespace thirdai::data {

class SequenceUnrolling final : public Transformation {
 public:
  SequenceUnrolling(std::string source_input_column,
                    std::string target_input_column,
                    std::string source_output_column,
                    std::string target_output_column)
      : _source_input_column(std::move(source_input_column)),
        _target_input_column(std::move(target_input_column)),
        _source_output_column(std::move(source_output_column)),
        _target_output_column(std::move(target_output_column)) {}

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _source_input_column;
  std::string _target_input_column;
  std::string _source_output_column;
  std::string _target_output_column;
};

}  // namespace thirdai::data