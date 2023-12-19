#pragma once

#include <smx/src/modules/Module.h>
#include <smx/src/optimizers/Optimizer.h>
#include <unordered_map>

namespace thirdai::smx {

class Adam final : public Optimizer {
 public:
  Adam(const std::vector<VariablePtr>& parameters, float lr, float beta_1 = 0.9,
       float beta_2 = 0.999, float eps = 1e-8);

  void step(VariablePtr& parameter) final;

 private:
  struct AdamInfo {
    DenseTensorPtr momentum;
    DenseTensorPtr velocity;
  };

  std::unordered_map<const Variable*, AdamInfo> _adam_parameters;

  float _lr;
  float _beta_1;
  float _beta_2;
  float _eps;
};

}  // namespace thirdai::smx