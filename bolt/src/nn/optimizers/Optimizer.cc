#include "Optimizer.h"
#include <bolt/src/nn/optimizers/Adam.h>
#include <bolt/src/nn/optimizers/SGD.h>
#include <stdexcept>

namespace thirdai::bolt {

std::unique_ptr<Optimizer> Optimizer::fromArchive(const ar::Archive& archive) {
  auto type = archive.str("type");

  if (type == Adam::type()) {
    return Adam::fromArchive(archive);
  }

  if (type == SGD::type()) {
    return SGD::fromArchive(archive);
  }

  throw std::invalid_argument("Invalid optimizer type specifier '" + type +
                              "' in fromArchive.");
}

std::shared_ptr<OptimizerFactory> OptimizerFactory::fromArchive(
    const ar::Archive& archive) {
  auto type = archive.str("type");

  if (type == Adam::type()) {
    return AdamFactory::fromArchive(archive);
  }

  if (type == SGD::type()) {
    return SGDFactory::fromArchive(archive);
  }

  throw std::invalid_argument("Invalid optimizer type specifier '" + type +
                              "' in fromArchive.");
}

}  // namespace thirdai::bolt