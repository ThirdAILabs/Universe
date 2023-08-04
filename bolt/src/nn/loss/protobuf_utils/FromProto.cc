#include "FromProto.h"
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <stdexcept>

namespace thirdai::bolt::nn::loss {

LossPtr fromProto(
    const proto::bolt::Loss& loss_proto,
    const std::unordered_map<std::string, autograd::ComputationPtr>&
        computations) {
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

}  // namespace thirdai::bolt::nn::loss