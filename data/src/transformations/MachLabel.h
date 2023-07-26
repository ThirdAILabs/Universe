#pragma once

#include <cereal/types/base_class.hpp>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/mach/MachIndex.h>

namespace thirdai::data {

using dataset::mach::MachIndexPtr;

class MachLabel final : public Transformation {
 public:
  MachLabel(std::string input_column, std::string output_column);

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _input_column;
  std::string _output_column;

  MachLabel() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data