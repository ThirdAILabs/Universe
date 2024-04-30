#pragma once

#include <smx/src/modules/Module.h>
#include <smx/src/optimizers/Optimizer.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/MaskedTensor.h>
#include <unordered_map>

namespace thirdai::smx {

class Adam final : public Optimizer {
 public:
  Adam(const std::vector<VariablePtr>& parameters, float lr, float beta_1 = 0.9,
       float beta_2 = 0.999, float eps = 1e-7,
       bool allow_sparse_updates = true);

  void update(VariablePtr& parameter) final;

  void updateLr(float lr) { _lr = lr; }

 private:
  void updateDense(const DenseTensorPtr& param, const DenseTensorPtr& grad,
                   const DenseTensorPtr& momentum,
                   const DenseTensorPtr& velocity);

  void updateSparse(const DenseTensorPtr& param, const MaskedTensorPtr& grad,
                    const DenseTensorPtr& momentum,
                    const DenseTensorPtr& velocity);

  struct AdamInfo {
    DenseTensorPtr momentum;
    DenseTensorPtr velocity;
  };

  std::unordered_map<const Variable*, AdamInfo> _adam_parameters;

  float _lr;
  float _beta_1;
  float _beta_2;
  float _eps;

  bool _allow_sparse_updates = true;
};

}  // namespace thirdai::smx