#pragma once

#include <data/src/transformations/Transformation.h>
namespace thirdai::data {

class AddMachRlhfSamples final : public Transformation {
 public:
  AddMachRlhfSamples(std::string input_indices_column,
                     std::string input_values_column, std::string label_column,
                     std::string mach_buckets_column);

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _input_indices_column;
  std::string _input_values_column;
  std::string _label_column;
  std::string _mach_buckets_column;

  // For cereal
  AddMachRlhfSamples() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data