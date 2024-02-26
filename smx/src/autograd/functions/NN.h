#pragma once

#include <smx/src/autograd/Variable.h>

namespace thirdai::smx {

VariablePtr embedding(const VariablePtr& indices, const VariablePtr& embs,
                      const VariablePtr& bias);

}  // namespace thirdai::smx