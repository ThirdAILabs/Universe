#include "Conversions.h"
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

proto::bolt::ActivationFunction activationToProto(
    ActivationFunction activation) {
  switch (activation) {
    case ActivationFunction::ReLU:
      return proto::bolt::ActivationFunction::RELU;
    case ActivationFunction::Softmax:
      return proto::bolt::ActivationFunction::SOFTMAX;
    case ActivationFunction::Sigmoid:
      return proto::bolt::ActivationFunction::SIGMOID;
    case ActivationFunction::Tanh:
      return proto::bolt::ActivationFunction::TANH;
    case ActivationFunction::Linear:
      return proto::bolt::ActivationFunction::LINEAR;
  }
}

ActivationFunction activationFromProto(
    proto::bolt::ActivationFunction activation) {
  switch (activation) {
    case proto::bolt::ActivationFunction::RELU:
      return ActivationFunction::ReLU;
    case proto::bolt::ActivationFunction::SOFTMAX:
      return ActivationFunction::Softmax;
    case proto::bolt::ActivationFunction::SIGMOID:
      return ActivationFunction::Sigmoid;
    case proto::bolt::ActivationFunction::TANH:
      return ActivationFunction::Tanh;
    case proto::bolt::ActivationFunction::LINEAR:
      return ActivationFunction::Linear;
    default:
      throw std::invalid_argument("Invalid activation function in fromProto.");
  }
}

proto::bolt::EmbeddingReduction reductionToProto(
    EmbeddingReductionType reduction) {
  switch (reduction) {
    case EmbeddingReductionType::CONCATENATION:
      return proto::bolt::EmbeddingReduction::CONCAT;
    case EmbeddingReductionType::SUM:
      return proto::bolt::EmbeddingReduction::SUM;
    case EmbeddingReductionType::AVERAGE:
      return proto::bolt::EmbeddingReduction::AVG;
  }
}

EmbeddingReductionType reductionFromProto(
    proto::bolt::EmbeddingReduction reduction) {
  switch (reduction) {
    case proto::bolt::EmbeddingReduction::CONCAT:
      return EmbeddingReductionType::CONCATENATION;
    case proto::bolt::EmbeddingReduction::SUM:
      return EmbeddingReductionType::SUM;
    case proto::bolt::EmbeddingReduction::AVG:
      return EmbeddingReductionType::AVERAGE;
    default:
      throw std::invalid_argument("Invalid reduction type in fromProto.");
  }
}

proto::bolt::Parameter* parametersToProto(
    const std::vector<float>& parameters) {
  proto::bolt::Parameter* proto = new proto::bolt::Parameter();
  proto->mutable_data()->Assign(parameters.begin(), parameters.end());
  return proto;
}

std::vector<float> parametersFromProto(const proto::bolt::Parameter& proto) {
  return {proto.data().begin(), proto.data().end()};
}

// This is a temporary method. It will be replaced when the optimizer PR merges.
// Some of these fields are to ensure that things serialized before the
// Optimizer PR can be loaded with the new design. For example beta1/beta2
// become optimizer parameters instead of global constants, so they are
// serialized here.
proto::bolt::Optimizer* optimizerToProto(const AdamOptimizer& optimizer,
                                         size_t rows, size_t cols) {
  if (optimizer.momentum.size() != (rows * cols)) {
    throw std::runtime_error("Rows and columns do not match optimizer size.");
  }
  proto::bolt::Optimizer* proto_opt = new proto::bolt::Optimizer();

  auto* adam = proto_opt->mutable_adam();

  adam->set_allocated_momentum(parametersToProto(optimizer.momentum));
  adam->set_allocated_velocity(parametersToProto(optimizer.velocity));

  adam->set_rows(rows);
  adam->set_cols(cols);

  adam->set_beta1(BETA1);
  adam->set_beta2(BETA2);
  adam->set_eps(EPS);

  return proto_opt;
}

AdamOptimizer optimizerFromProto(const proto::bolt::Optimizer& opt_proto) {
  if (!opt_proto.has_adam()) {
    throw std::invalid_argument("Expected adam optimizer.");
  }

  AdamOptimizer opt(0);

  opt.momentum = parametersFromProto(opt_proto.adam().momentum());
  opt.velocity = parametersFromProto(opt_proto.adam().velocity());

  if (opt.momentum.size() != opt.velocity.size()) {
    throw std::runtime_error(
        "Expected momentum and velocity to have the same size.");
  }

  opt.gradients.assign(opt.momentum.size(), 0.0);

  return opt;
}

}  // namespace thirdai::bolt::nn::ops