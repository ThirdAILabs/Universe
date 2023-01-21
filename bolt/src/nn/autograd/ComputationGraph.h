#pragma once

#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt::nn::autograd {

ComputationList getComputationOrder(const ComputationList& outputs);

}  // namespace thirdai::bolt::nn::autograd