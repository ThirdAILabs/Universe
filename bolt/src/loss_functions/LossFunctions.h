#pragma once

#include <bolt/src/layers/BoltVector.h>

namespace thirdai::bolt {

template <uint32_t SCALE> 
void computeDenseLoss(BoltVector&, const BoltVector&, uint32_t);

template <uint32_t SCALE> 
void computeSparseLoss(BoltVector&, const BoltVector&, uint32_t);

class SparseCategoricalCrossEntropyLoss {
 public:
  void operator()(BoltVector& output, const BoltVector& labels, uint32_t batch_size) const {
    if (output.isDense()) {
        computeDenseLoss<1>(output, labels, batch_size);
    } else {
      computeSparseLoss<1>(output, labels, batch_size);
    }
  }
};


class MeanSquaredError {
 public:
  void operator()(BoltVector& output, const BoltVector& labels, uint32_t batch_size) const {
    if (output.isDense()) {
        computeDenseLoss<2>(output, labels, batch_size);
    } else {
      computeSparseLoss<2>(output, labels, batch_size);
    }
  }
};

}  // namespace thirdai::bolt
