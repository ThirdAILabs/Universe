#pragma once

#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/Op.h>
#include <proto/loss.pb.h>
#include <string>
#include <unordered_map>

namespace thirdai::bolt::nn::loss {

LossPtr fromProto(
    const proto::bolt::Loss& loss_proto,
    const std::unordered_map<std::string, autograd::ComputationPtr>&
        computations);

}  // namespace thirdai::bolt::nn::loss