#include "Layer.h"
#include <dataset/src/Dataset.h>

namespace thirdai::bolt {

template <>
VectorState VectorState::makeInputStateFromBatch<dataset::SparseBatch>(
    const dataset::SparseBatch& input_batch, uint32_t i) {
  return VectorState::makeSparseInputState(
      input_batch[i]._indices, input_batch[i]._values, input_batch[i].length());
}

template <>
VectorState VectorState::makeInputStateFromBatch<dataset::DenseBatch>(
    const dataset::DenseBatch& input_batch, uint32_t i) {
  return VectorState::makeDenseInputState(input_batch[i]._values,
                                          input_batch[i].dim());
}

template <typename BATCH_T>
VectorState VectorState::makeInputStateFromBatch(const BATCH_T& input_batch,
                                                 uint32_t i) {
  throw std::invalid_argument("Input batch type is not supported.");
}

}  // namespace thirdai::bolt