#include "Adam.h"
#include <smx/src/tensor/Init.h>
#include <iostream>

namespace thirdai::smx {

Adam::Adam(const std::vector<VariablePtr>& parameters, float lr, float beta_1,
           float beta_2, float eps)
    : Optimizer(parameters),
      _lr(lr),
      _beta_1(beta_1),
      _beta_2(beta_2),
      _eps(eps) {
  for (const auto& param : parameters) {
    _adam_parameters[param.get()] = {
        /*momentum=*/zeros(param->tensor()->shape()),
        /*velocity=*/zeros(param->tensor()->shape()),
    };
  }
}

void Adam::apply(VariablePtr& parameter) {
  float b1_corrected = 1 - pow(_beta_1, _n_steps);
  float b2_corrected = 1 - pow(_beta_2, _n_steps);

  auto [momentum, velocity] = _adam_parameters.at(parameter.get());

  auto param = asDense(parameter->tensor())->eigenArray<float>();
  auto grad = asDense(parameter->grad())->eigenArray<float>();
  auto mom = asDense(momentum)->eigenArray<float>();
  auto vel = asDense(velocity)->eigenArray<float>();

  mom = _beta_1 * mom + (1 - _beta_1) * grad;
  vel = _beta_2 * vel + (1 - _beta_2) * grad.square();

  param += (_lr / b1_corrected) * mom / ((vel / b2_corrected).sqrt() + _eps);

  // TODO(Nicholas): Should we add a check for NaNs here?
}

}  // namespace thirdai::smx