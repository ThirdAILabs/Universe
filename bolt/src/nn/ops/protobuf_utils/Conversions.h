#pragma once

#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/layers/Optimizer.h>
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

proto::bolt::Parameter* parametersToProto(const std::vector<float>& parameters);

std::vector<float> parametersFromProto(const proto::bolt::Parameter& proto);

proto::bolt::Optimizer* optimizerToProto(const AdamOptimizer& optimizer,
                                         size_t rows, size_t cols);

AdamOptimizer optimizerFromProto(const proto::bolt::Optimizer& opt_proto);

}  // namespace thirdai::bolt