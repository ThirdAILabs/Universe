#include "FromProto.h"
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/RobeZ.h>
#include <bolt/src/nn/ops/Tanh.h>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

template <typename OP_TYPE>
OpApplyFunc unaryApplyFunc() {
  auto apply_func = [](const ops::OpPtr& op,
                       const autograd::ComputationList& inputs) {
    auto concrete_op = std::dynamic_pointer_cast<OP_TYPE>(op);
    if (!concrete_op) {
      throw std::runtime_error("Op type mismatch in apply func.");
    }

    if (inputs.size() != 1) {
      throw std::invalid_argument("Expected 1 input to op '" + op->name() +
                                  "'.");
    }

    return concrete_op->apply(inputs.at(0));
  };

  return apply_func;
}

autograd::ComputationPtr concatenateApplyFunc(
    const ops::OpPtr& op, const autograd::ComputationList& inputs) {
  auto concat_op = std::dynamic_pointer_cast<Concatenate>(op);
  if (!concat_op) {
    throw std::runtime_error("Op type mismatch in apply func.");
  }

  return concat_op->apply(inputs);
}

std::pair<OpPtr, OpApplyFunc> fromProto(const proto::bolt::Op& op_proto) {
  const std::string& name = op_proto.name();

  OpPtr op;
  switch (op_proto.type_case()) {
    case proto::bolt::Op::kFullyConnected:
      op = FullyConnected::fromProto(name, op_proto.fully_connected());
      return {op, unaryApplyFunc<FullyConnected>()};
    case proto::bolt::Op::kEmbedding:
      op = Embedding::fromProto(name, op_proto.embedding());
      return {op, unaryApplyFunc<Embedding>()};
    case proto::bolt::Op::kRobez:
      op = RobeZ::fromProto(name, op_proto.robez());
      return {op, unaryApplyFunc<RobeZ>()};
    case proto::bolt::Op::kConcatenate:
      op = Concatenate::fromProto(name, op_proto.concatenate());
      return {op, concatenateApplyFunc};
    case proto::bolt::Op::kLayerNorm:
      op = LayerNorm::fromProto(name, op_proto.layer_norm());
      return {op, unaryApplyFunc<LayerNorm>()};
    case proto::bolt::Op::kTanh:
      op = Tanh::fromProto(name, op_proto.tanh());
      return {op, unaryApplyFunc<Tanh>()};

    case proto::bolt::Op::TYPE_NOT_SET:
      throw std::invalid_argument("Invalid op with TYPE_NOT_SET in fromProto.");
  }
}

}  // namespace thirdai::bolt::nn::ops