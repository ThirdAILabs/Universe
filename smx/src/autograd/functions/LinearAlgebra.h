#pragma once

#include <smx/src/autograd/Variable.h>

namespace thirdai::smx {

VariablePtr add(const VariablePtr& a, const VariablePtr& b);

VariablePtr linear(const VariablePtr& x, const VariablePtr& w,
                   const VariablePtr& b);

}  // namespace thirdai::smx