#pragma once

#include <data/src/transformations/Transformation.h>
#include <dataset/src/mach/MachIndex.h>
#include <memory>

namespace thirdai::data {

using dataset::mach::MachIndexPtr;

class MachLabel final : public Transformation {
 public:
  MachLabel(std::string input_column_name, std::string output_column_name);

  static std::shared_ptr<MachLabel> make(std::string input_column_name,
                                         std::string output_column_name) {
    return std::make_shared<MachLabel>(std::move(input_column_name),
                                       std::move(output_column_name));
  }

  explicit MachLabel(const proto::data::MachLabel& mach_label);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  proto::data::Transformation* toProto() const final;

 private:
  std::string _input_column_name;
  std::string _output_column_name;
};

}  // namespace thirdai::data