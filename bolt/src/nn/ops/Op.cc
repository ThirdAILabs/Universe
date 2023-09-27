#include "Op.h"
#include <bolt/src/nn/ops/Activation.h>
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/CosineSimilarity.h>
#include <bolt/src/nn/ops/DotProduct.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/RobeZ.h>
#include <stdexcept>

namespace thirdai::bolt {

std::shared_ptr<Op> Op::fromProto(const proto::bolt::Op& op_proto,
                                  DeserializedParameters& parameters) {
  const std::string& name = op_proto.name();

  OpPtr op;
  switch (op_proto.type_case()) {
    case proto::bolt::Op::kFullyConnected:
      return FullyConnected::fromProto(name, op_proto.fully_connected(),
                                       parameters);

    case proto::bolt::Op::kEmbedding:
      return Embedding::fromProto(name, op_proto.embedding(), parameters);

    case proto::bolt::Op::kRobez:
      return RobeZ::fromProto(name, op_proto.robez(), parameters);

    case proto::bolt::Op::kConcatenate:
      return Concatenate::fromProto(name, op_proto.concatenate());

    case proto::bolt::Op::kLayerNorm:
      return LayerNorm::fromProto(name, op_proto.layer_norm(), parameters);

    case proto::bolt::Op::kDotProduct:
      return DotProduct::fromProto(name, op_proto.dot_product());

    case proto::bolt::Op::kCosineSimilarity:
      return CosineSimilarity::fromProto(name, op_proto.cosine_similarity());

    case proto::bolt::Op::kActivation:
      return activationOpFromProto(name, op_proto.activation());

    default:
      throw std::invalid_argument("Invalid op type in fromProto.");
  }
}

}  // namespace thirdai::bolt