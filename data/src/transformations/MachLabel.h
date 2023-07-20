#pragma once

#include <data/src/transformations/Transformation.h>
#include <dataset/src/mach/MachIndex.h>

namespace thirdai::data {

using dataset::mach::MachIndexPtr;

class MachLabel final : public Transformation {
 public:
  MachLabel(std::string input_column, std::string output_column,
            MachIndexPtr index);

  ColumnMap apply(ColumnMap columns) const final;

  MachIndexPtr index() const { return _index; }

  void setIndex(const MachIndexPtr& index);

 private:
  MachIndexPtr _index;

  std::string _input_column;
  std::string _output_column;
};

}  // namespace thirdai::data