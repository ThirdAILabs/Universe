#pragma once

#include <dataset/src/data_pipeline/ColumnMap.h>
#include <memory>

namespace thirdai::dataset {

class Transformation {
 public:
  virtual void apply(ColumnMap& columns) = 0;

  virtual ~Transformation() = default;
};

using TransformationPtr = std::shared_ptr<Transformation>;

}  // namespace thirdai::dataset