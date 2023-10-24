#pragma once

#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/layers/Optimizer.h>
#include <bolt/src/nn/ops/protobuf_utils/SerializedParameters.h>
#include <proto/ops.pb.h>
#include <proto/optimizers.pb.h>
#include <proto/parameter.pb.h>

namespace thirdai::bolt {

proto::bolt::ActivationFunction activationToProto(
    ActivationFunction activation);

ActivationFunction activationFromProto(
    proto::bolt::ActivationFunction activation);

proto::bolt::EmbeddingReduction reductionToProto(
    EmbeddingReductionType reduction);

EmbeddingReductionType reductionFromProto(
    proto::bolt::EmbeddingReduction reduction);

proto::bolt::Parameter* parametersToProto(const std::string& name);

std::vector<float> parametersFromProto(const proto::bolt::Parameter& parameter,
                                       DeserializedParameters& parameters);

proto::bolt::Optimizer* optimizerToProto(const std::string& param_name,
                                         size_t rows, size_t cols);

void addOptimizerParameters(const AdamOptimizer& optimizer,
                            const std::string& param_name,
                            SerializableParameters& parameters);

AdamOptimizer optimizerFromProto(const proto::bolt::Optimizer& opt_proto,
                                 DeserializedParameters& parameters);

}  // namespace thirdai::bolt