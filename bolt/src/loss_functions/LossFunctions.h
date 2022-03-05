#pragma once

#include <bolt/src/layers/BoltVector.h>

namespace thirdai::bolt {

class CategoricalCrossEntropyLoss {
 public:
  void operator()(BoltVector& output, const BoltVector& labels,
                  uint32_t batch_size) const;
};

class MeanSquaredError {
 public:
  void operator()(BoltVector& output, const BoltVector& labels,
                  uint32_t batch_size) const;
};

}  // namespace thirdai::bolt
