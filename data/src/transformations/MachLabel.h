#pragma once

#include <cereal/access.hpp>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/mach/MachIndex.h>
#include <memory>

namespace thirdai::data {

using dataset::mach::MachIndexPtr;

class MachLabel final : public Transformation {
 public:
  MachLabel(std::string input_column_name, std::string output_column_name);

  static auto make(std::string input_column_name,
                   std::string output_column_name) {
    return std::make_shared<MachLabel>(std::move(input_column_name),
                                       std::move(output_column_name));
  }

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _input_column_name;
  std::string _output_column_name;

  MachLabel() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data