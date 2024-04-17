#pragma once

#include <data/src/transformations/Transformation.h>

namespace thirdai::automl {

class PretrainedAugmentation {
 public:
  virtual data::TransformationPtr transformation(
      const std::string& input_col, const std::string& output_col) const = 0;
};

}  // namespace thirdai::automl