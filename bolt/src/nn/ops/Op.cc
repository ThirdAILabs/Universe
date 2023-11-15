#include "Op.h"
#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/nn/ops/Activation.h>
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/CosineSimilarity.h>
#include <bolt/src/nn/ops/DlrmAttention.h>
#include <bolt/src/nn/ops/DotProduct.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/PatchEmbedding.h>
#include <bolt/src/nn/ops/PatchSum.h>
#include <bolt/src/nn/ops/RobeZ.h>
#include <bolt/src/nn/ops/Switch.h>
#include <archive/src/Archive.h>
#include <archive/src/ArchiveMap.h>
#include <archive/src/ParameterReference.h>
#include <stdexcept>
#include <string>

namespace thirdai::bolt {

std::shared_ptr<Op> Op::fromArchive(const ar::Archive& archive) {
  std::string type = archive.getAs<std::string>("type");

  if (type == Activation<ReluImpl>::type()) {
    return activationOpFromArchive(archive);
  }
  if (type == Concatenate::type()) {
    return Concatenate::fromArchive(archive);
  }
  if (type == CosineSimilarity::type()) {
    return CosineSimilarity::fromArchive(archive);
  }
  if (type == DlrmAttention::type()) {
    return DlrmAttention::fromArchive(archive);
  }
  if (type == DotProduct::type()) {
    return DotProduct::fromArchive(archive);
  }
  if (type == Embedding::type()) {
    return Embedding::fromArchive(archive);
  }
  if (type == FullyConnected::type()) {
    return FullyConnected::fromArchive(archive);
  }
  if (type == LayerNorm::type()) {
    return LayerNorm::fromArchive(archive);
  }
  if (type == PatchEmbedding::type()) {
    return PatchEmbedding::fromArchive(archive);
  }
  if (type == PatchSum::type()) {
    return PatchSum::fromArchive(archive);
  }
  if (type == RobeZ::type()) {
    return RobeZ::fromArchive(archive);
  }
  if (type == Switch::type()) {
    return Switch::fromArchive(archive);
  }
  throw std::invalid_argument("Invalid op type '" + type + "'.");
}

// This is a temporary method. It will be replaced when the optimizer PR merges.
// Some of these fields are to ensure that things serialized before the
// Optimizer PR can be loaded with the new design. For example beta1/beta2
// become optimizer parameters instead of global constants, so they are
// serialized here. Rows/cols will be used by the optimizers, but are currently
// not required because the optimizer is hard coded into the ops/layers.
ar::ConstArchivePtr optimizerToArchive(const AdamOptimizer& optimizer,
                                       const std::shared_ptr<const Op>& op,
                                       size_t rows, size_t cols) {
  auto map = ar::ArchiveMap::make();

  map->set("type", ar::str("adam"));

  map->set("momentum", ar::ParameterReference::make(optimizer.momentum, op));
  map->set("velocity", ar::ParameterReference::make(optimizer.velocity, op));

  map->set("rows", ar::u64(rows));
  map->set("cols", ar::u64(cols));

  map->set("beta1", ar::f32(BETA1));
  map->set("beta2", ar::f32(BETA2));
  map->set("eps", ar::f32(EPS));

  return map;
}

AdamOptimizer optimizerFromArchive(const ar::Archive& archive) {
  if (archive.getAs<ar::Str>("type") != "adam") {
    throw std::invalid_argument("Expected optimizer to have type 'adam'.");
  }
  AdamOptimizer optimizer;

  optimizer.momentum = archive.get("momentum")->param().moveLoadedParameter();
  optimizer.velocity = archive.get("velocity")->param().moveLoadedParameter();

  if (optimizer.momentum.size() != optimizer.velocity.size()) {
    throw std::runtime_error(
        "Expected momentum and velocity to have the same size.");
  }

  optimizer.gradients.assign(optimizer.momentum.size(), 0.0);

  return optimizer;
}

}  // namespace thirdai::bolt