#include "FromProto.h"
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/RobeZ.h>
#include <bolt/src/nn/ops/Tanh.h>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

OpPtr fromProto(const proto::bolt::Op& op_proto) {
  const std::string& name = op_proto.name();

  OpPtr op;
  switch (op_proto.type_case()) {
    case proto::bolt::Op::kFullyConnected:
      return FullyConnected::fromProto(name, op_proto.fully_connected());

    case proto::bolt::Op::kEmbedding:
      return Embedding::fromProto(name, op_proto.embedding());

    case proto::bolt::Op::kRobez:
      return RobeZ::fromProto(name, op_proto.robez());

    case proto::bolt::Op::kConcatenate:
      return Concatenate::fromProto(name, op_proto.concatenate());

    case proto::bolt::Op::kLayerNorm:
      return LayerNorm::fromProto(name, op_proto.layer_norm());

    case proto::bolt::Op::kTanh:
      return Tanh::fromProto(name, op_proto.tanh());

    case proto::bolt::Op::TYPE_NOT_SET:
      throw std::invalid_argument("Invalid op with TYPE_NOT_SET in fromProto.");
  }
}

}  // namespace thirdai::bolt::nn::ops