#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <proto/ops.pb.h>

namespace thirdai::bolt::nn::ops {

using OpApplyFunc = std::function<autograd::ComputationPtr(
    const ops::OpPtr& op, const autograd::ComputationList& inputs)>;

std::pair<OpPtr, OpApplyFunc> fromProto(const proto::bolt::Op& op_proto);

}  // namespace thirdai::bolt::nn::ops