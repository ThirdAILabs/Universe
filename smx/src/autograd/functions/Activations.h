#pragma once

#include <smx/src/autograd/Variable.h>

namespace thirdai::smx {

VariablePtr relu(const VariablePtr& input);

VariablePtr tanh(const VariablePtr& input);

VariablePtr sigmoid(const VariablePtr& input);

VariablePtr softmax(const VariablePtr& input);

}  // namespace thirdai::smx