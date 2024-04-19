#pragma once

#include <data/src/transformations/Transformation.h>

namespace thirdai::data {
  class ConcatTokenArrayColumn final : public Transformation {
    public:
     ConcatTokenArrayColumn(std::string input_indices_column, std::string input_values_column,
                            std::string second_input_column, std::string output_indices_column, 
                            std::string output_value_column);

  explicit ConcatTokenArrayColumn(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "concat_token_array_column"; }

 private:

  std::string _input_indices_column; 
  std::string _input_values_column;
  std::string _second_input_column; 
  std::string _output_indices_column; 
  std::string _output_value_column;

  ConcatTokenArrayColumn() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
  };
}  // namespace thirdai::data