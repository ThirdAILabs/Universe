#pragma once

#include <smx/src/autograd/Variable.h>

namespace thirdai::smx {

VariablePtr embedding(const VariablePtr& indices, const VariablePtr& embs,
                      bool reduce_mean);

}  // namespace thirdai::smx