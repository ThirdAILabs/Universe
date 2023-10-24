#include "Conversions.h"
#include <google/protobuf/repeated_field.h>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt {

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
    default:
      throw std::invalid_argument("Invalid activation function in toProto.");
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
    default:
      throw std::invalid_argument("Invalid reduction type in toProto.");
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

proto::bolt::Parameter* parametersToProto(const std::string& name) {
  proto::bolt::Parameter* proto = new proto::bolt::Parameter();
  proto->set_name(name);

  return proto;
}

std::vector<float> parametersFromProto(const proto::bolt::Parameter& parameter,
                                       DeserializedParameters& parameters) {
  if (!parameters.count(parameter.name())) {
#if THIRDAI_EXPOSE_ALL
    throw std::invalid_argument("Unable to locate parameter '" +
                                parameter.name() + "' when loading model.");
#else
    throw std::invalid_argument(
        "Encountered missing parameter when loading model.");
#endif
  }

  std::vector<float> data = std::move(parameters.at(parameter.name()));
  parameters.erase(parameter.name());

  return data;
}

// This is a temporary method. It will be replaced when the optimizer PR merges.
// Some of these fields are to ensure that things serialized before the
// Optimizer PR can be loaded with the new design. For example beta1/beta2
// become optimizer parameters instead of global constants, so they are
// serialized here. Rows/cols will be used by the optimizers, but are currently
// not required because the optimizer is hard coded into the ops/layers.
proto::bolt::Optimizer* optimizerToProto(const std::string& param_name,
                                         size_t rows, size_t cols) {
  proto::bolt::Optimizer* proto_opt = new proto::bolt::Optimizer();

  auto* adam = proto_opt->mutable_adam();

  adam->set_allocated_momentum(parametersToProto(param_name + "_momentum"));
  adam->set_allocated_velocity(parametersToProto(param_name + "_velocity"));

  adam->set_rows(rows);
  adam->set_cols(cols);

  adam->set_beta1(BETA1);
  adam->set_beta2(BETA2);
  adam->set_eps(EPS);

  return proto_opt;
}

void addOptimizerParameters(const AdamOptimizer& optimizer,
                            const std::string& param_name,
                            SerializableParameters& parameters) {
  parameters.emplace_back(param_name + "_momentum", &optimizer.momentum);
  parameters.emplace_back(param_name + "_velocity", &optimizer.velocity);
}

AdamOptimizer optimizerFromProto(const proto::bolt::Optimizer& opt_proto,
                                 DeserializedParameters& parameters) {
  if (!opt_proto.has_adam()) {
    throw std::invalid_argument("Expected adam optimizer.");
  }

  AdamOptimizer opt(0);

  opt.momentum = parametersFromProto(opt_proto.adam().momentum(), parameters);
  opt.velocity = parametersFromProto(opt_proto.adam().velocity(), parameters);

  if (opt.momentum.size() != opt.velocity.size()) {
    throw std::runtime_error(
        "Expected momentum and velocity to have the same size.");
  }

  opt.gradients.assign(opt.momentum.size(), 0.0);

  return opt;
}

}  // namespace thirdai::bolt