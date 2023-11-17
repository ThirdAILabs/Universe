#pragma once

#include <smx/src/autograd/Variable.h>
#include <vector>

namespace thirdai::smx {

class Module {
 public:
  virtual std::vector<VariablePtr> forward(
      const std::vector<VariablePtr>& inputs) = 0;

  virtual std::vector<VariablePtr> parameters() const = 0;
};

}  // namespace thirdai::smx