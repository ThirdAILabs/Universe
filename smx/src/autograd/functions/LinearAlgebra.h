#pragma once

#include <smx/src/autograd/Variable.h>

namespace thirdai::smx {

VariablePtr add(const VariablePtr& a, const VariablePtr& b);

VariablePtr linear(const VariablePtr& x, const VariablePtr& w,
                   const VariablePtr& b);

VariablePtr linear(const VariablePtr& x, const VariablePtr& w,
                   const VariablePtr& b, float sparsity,
                   const NeuronIndexPtr& neuron_index,
                   const VariablePtr& labels);

}  // namespace thirdai::smx