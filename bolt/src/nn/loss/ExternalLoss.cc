#include "ExternalLoss.h"
#include <cereal/types/base_class.hpp>
#include <stdexcept>

namespace thirdai::bolt {

void ExternalLoss::gradients(uint32_t index_in_batch,
                             uint32_t batch_size) const {
  if (_output->tensor()->batchSize() != batch_size ||
      _external_gradients->tensor()->batchSize() != batch_size) {
    throw std::invalid_argument(
        "Mismatched batch size between outputs and external gradients.");
  }

  BoltVector& output = _output->tensor()->getVector(index_in_batch);
  const BoltVector& grad =
      _external_gradients->tensor()->getVector(index_in_batch);

  if (output.len != grad.len || output.isDense() != grad.isDense()) {
    throw std::invalid_argument(
        "Expected size and sparsity of external gradient to match that of the "
        "output.");
  }

  std::copy(grad.activations, grad.activations + grad.len, output.gradients);
}

float ExternalLoss::loss(uint32_t index_in_batch) const {
  (void)index_in_batch;
  throw std::invalid_argument(
      "Cannot compute loss for external loss function.");
}

template <class Archive>
void ExternalLoss::serialize(Archive& archive) {
  archive(cereal::base_class<Loss>(this), _output, _external_gradients);
}

}  // namespace thirdai::bolt