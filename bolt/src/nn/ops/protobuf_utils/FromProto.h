#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <proto/ops.pb.h>

namespace thirdai::bolt::nn::ops {

OpPtr fromProto(const proto::bolt::Op& op_proto);

}  // namespace thirdai::bolt::nn::ops