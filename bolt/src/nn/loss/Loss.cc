#include "Loss.h"
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/EuclideanContrastive.h>
#include <bolt/src/nn/loss/ExternalLoss.h>
#include <stdexcept>

namespace thirdai::bolt {

std::shared_ptr<Loss> Loss::fromArchive(
    const ar::Archive& archive,
    const std::unordered_map<std::string, ComputationPtr>& computations) {
  std::string type = archive.str("type");

  if (type == CategoricalCrossEntropy::type()) {
    return CategoricalCrossEntropy::fromArchive(archive, computations);
  }

  if (type == BinaryCrossEntropy::type()) {
    return BinaryCrossEntropy::fromArchive(archive, computations);
  }

  if (type == EuclideanContrastive::type()) {
    return EuclideanContrastive::fromArchive(archive, computations);
  }

  if (type == ExternalLoss::type()) {
    return ExternalLoss::fromArchive(archive, computations);
  }

  throw std::invalid_argument("Invalid loss type '" + type + "'.");
}

void assertLossType(const ar::Archive& archive,
                    const std::string& expected_type) {
  auto type = archive.str("type");
  if (type != expected_type) {
    throw std::invalid_argument("Expected loss type '" + expected_type +
                                "' but received type '" + type + "'.");
  }
}

}  // namespace thirdai::bolt