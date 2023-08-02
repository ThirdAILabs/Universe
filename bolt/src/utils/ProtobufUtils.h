#pragma once

#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/layers/Optimizer.h>
#include <proto/ops.pb.h>
#include <proto/optimizers.pb.h>
#include <proto/parameter.pb.h>
#include <stdexcept>

namespace thirdai::bolt::utils {

inline bolt_proto::ActivationFunction activationToProto(
    ActivationFunction activation) {
  switch (activation) {
    case ActivationFunction::ReLU:
      return bolt_proto::ActivationFunction::RELU;
    case ActivationFunction::Softmax:
      return bolt_proto::ActivationFunction::SOFTMAX;
    case ActivationFunction::Sigmoid:
      return bolt_proto::ActivationFunction::SIGMOID;
    case ActivationFunction::Tanh:
      return bolt_proto::ActivationFunction::TANH;
    case ActivationFunction::Linear:
      return bolt_proto::ActivationFunction::LINEAR;
  }
}

inline ActivationFunction activationFromProto(
    bolt_proto::ActivationFunction activation) {
  switch (activation) {
    case bolt_proto::ActivationFunction::RELU:
      return ActivationFunction::ReLU;
    case bolt_proto::ActivationFunction::SOFTMAX:
      return ActivationFunction::Softmax;
    case bolt_proto::ActivationFunction::SIGMOID:
      return ActivationFunction::Sigmoid;
    case bolt_proto::ActivationFunction::TANH:
      return ActivationFunction::Tanh;
    case bolt_proto::ActivationFunction::LINEAR:
      return ActivationFunction::Linear;
    default:
      throw std::invalid_argument("Invalid activation function in fromProto.");
  }
}

inline bolt_proto::Parameter* parametersToProto(
    const std::vector<float>& parameters) {
  bolt_proto::Parameter* proto = new bolt_proto::Parameter();
  proto->mutable_data()->Assign(parameters.begin(), parameters.end());
  return proto;
}

inline std::vector<float> parametersFromProto(
    const bolt_proto::Parameter& proto) {
  return {proto.data().begin(), proto.data().end()};
}

// This is a temporary method. It will be replaced when the optimizer PR merges.
// Some of these fields are to ensure that things serialized before the
// Optimizer PR can be loaded with the new design. For example beta1/beta2
// become optimizer parameters instead of global constants, so they are
// serialized here.
inline bolt_proto::Optimizer* optimizerToProto(const AdamOptimizer& optimizer,
                                               size_t rows, size_t cols) {
  bolt_proto::Optimizer* proto_opt = new bolt_proto::Optimizer();

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

}  // namespace thirdai::bolt::utils