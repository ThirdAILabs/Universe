#include "Loss.h"
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <stdexcept>

namespace thirdai::bolt {

std::shared_ptr<Loss> Loss::fromProto(
    const proto::bolt::Loss& loss_proto,
    const std::unordered_map<std::string, ComputationPtr>& computations) {
  switch (loss_proto.loss_case()) {
    case proto::bolt::Loss::kCategoricalCrossEntropy:
      return CategoricalCrossEntropy::make(
          computations.at(loss_proto.categorical_cross_entropy().output()),
          computations.at(loss_proto.categorical_cross_entropy().labels()));
    case proto::bolt::Loss::kBinaryCrossEntropy:
      return BinaryCrossEntropy::make(
          computations.at(loss_proto.binary_cross_entropy().output()),
          computations.at(loss_proto.binary_cross_entropy().labels()));
    default:
      throw std::invalid_argument("Invalid loss in fromProto.");
  }
}

}  // namespace thirdai::bolt