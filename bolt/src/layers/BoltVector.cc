#include "BoltVector.h"
#include <dataset/src/Dataset.h>

namespace thirdai::bolt {

template <>
BoltVector BoltVector::makeInputStateFromBatch<dataset::SparseBatch>(
    const dataset::SparseBatch& input_batch, uint32_t i) {
  return BoltVector::makeSparseInputState(
      input_batch[i]._indices, input_batch[i]._values, input_batch[i].length());
}

template <>
BoltVector BoltVector::makeInputStateFromBatch<dataset::DenseBatch>(
    const dataset::DenseBatch& input_batch, uint32_t i) {
  return BoltVector::makeDenseInputState(input_batch[i]._values,
                                         input_batch[i].dim());
}

template <typename BATCH_T>
BoltVector BoltVector::makeInputStateFromBatch(const BATCH_T& input_batch,
                                               uint32_t i) {
  (void)input_batch;
  (void)i;
  throw std::invalid_argument("Input batch type is not supported.");
}

}  // namespace thirdai::bolt