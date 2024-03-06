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
#include <bolt/src/nn/ops/WeightedSum.h>
#include <bolt/src/nn/ops/QuantileMixing.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <archive/src/ParameterReference.h>
#include <stdexcept>
#include <string>

namespace thirdai::bolt {

std::shared_ptr<Op> Op::fromArchive(const ar::Archive& archive) {
  std::string type = archive.str("type");

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
  if (type == WeightedSum::type()) {
    return WeightedSum::fromArchive(archive);
  }
  if (type == QuantileMixing::type()) {
    return QuantileMixing::fromArchive(archive);
  }
  throw std::invalid_argument("Invalid op type '" + type + "'.");
}

void assertOpType(const ar::Archive& archive,
                  const std::string& expected_type) {
  auto type = archive.str("type");
  if (type != expected_type) {
    throw std::invalid_argument("Expected op type '" + expected_type +
                                "' but received type '" + type + "'.");
  }
}

}  // namespace thirdai::bolt