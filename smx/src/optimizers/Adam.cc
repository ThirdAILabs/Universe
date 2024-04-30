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

constexpr size_t N_CHUNKS = 256;
constexpr size_t MIN_CHUNKSIZE = 4096;

void Adam::update(VariablePtr& parameter) {
  auto param = dense(parameter->tensor());

  auto [momentum, velocity] = _adam_parameters.at(parameter.get());

  CHECK(param->shape() == parameter->grad()->shape(),
        "Param and grad must have same shape.");
  CHECK(param->shape() == momentum->shape(),
        "Param and momentum must have same shape.");
  CHECK(param->shape() == velocity->shape(),
        "Param and velocity must have same shape.");

  if (parameter->grad()->isMasked()) {
    updateSparse(param, masked(parameter->grad()), momentum, velocity);
  } else {
    updateDense(param, dense(parameter->grad()), momentum, velocity);
  }
}

void Adam::updateDense(const DenseTensorPtr& param, const DenseTensorPtr& grad,
                       const DenseTensorPtr& momentum,
                       const DenseTensorPtr& velocity) {
  float b1_corrected = 1 - pow(_beta_1, _n_steps);
  float b2_corrected = 1 - pow(_beta_2, _n_steps);

  auto* param_ptr = param->data<float>();
  auto* grad_ptr = grad->data<float>();
  auto* mom_ptr = momentum->data<float>();
  auto* vel_ptr = velocity->data<float>();

  const size_t param_size = param->size();
  const size_t chunk_size =
      std::max((param_size + N_CHUNKS - 1) / N_CHUNKS, MIN_CHUNKSIZE);

#pragma omp parallel for default(none)                                    \
    shared(param_size, chunk_size, param_ptr, grad_ptr, mom_ptr, vel_ptr, \
           b1_corrected, b2_corrected) if (param_size > chunk_size)
  for (size_t start = 0; start < param_size; start += chunk_size) {
    size_t end = std::min(start + chunk_size, param_size);

    size_t len = end - start;
    EigenArray<float> param_arr(param_ptr + start, len);
    EigenArray<float> grad_arr(grad_ptr + start, len);
    EigenArray<float> mom_arr(mom_ptr + start, len);
    EigenArray<float> vel_arr(vel_ptr + start, len);

    mom_arr = _beta_1 * mom_arr + (1 - _beta_1) * grad_arr;
    vel_arr = _beta_2 * vel_arr + (1 - _beta_2) * grad_arr.square();

    param_arr += (_lr / b1_corrected) * mom_arr /
                 ((vel_arr / b2_corrected).sqrt() + _eps);

    // TODO(Nicholas): Should we add a check for NaNs here, or just let them
    // show up in the loss?
  }
}

void Adam::updateSparse(const DenseTensorPtr& param,
                        const MaskedTensorPtr& grad,
                        const DenseTensorPtr& momentum,
                        const DenseTensorPtr& velocity) {
  float b1_corrected = 1 - pow(_beta_1, _n_steps);
  float b2_corrected = 1 - pow(_beta_2, _n_steps);

  auto* param_ptr = param->data<float>();
  auto* grad_ptr = grad->data<float>();
  auto* mom_ptr = momentum->data<float>();
  auto* vel_ptr = velocity->data<float>();

  size_t rows = param->shape(0);
  size_t cols = param->shape(1);

  const auto& rows_used = grad->mask();

#pragma omp parallel for default(none),                                  \
    shared(rows, cols, param_ptr, grad_ptr, mom_ptr, vel_ptr, rows_used, \
           b1_corrected, b2_corrected)
  for (size_t row = 0; row < rows; row++) {
    if (!rows_used[row]) {
      continue;
    }

    size_t offset = row * cols;

    EigenArray<float> param_arr(param_ptr + offset, cols);
    EigenArray<float> grad_arr(grad_ptr + offset, cols);
    EigenArray<float> mom_arr(mom_ptr + offset, cols);
    EigenArray<float> vel_arr(vel_ptr + offset, cols);

    mom_arr = _beta_1 * mom_arr + (1 - _beta_1) * grad_arr;
    vel_arr = _beta_2 * vel_arr + (1 - _beta_2) * grad_arr.square();

    param_arr += (_lr / b1_corrected) * mom_arr /
                 ((vel_arr / b2_corrected).sqrt() + _eps);
  }
}

}  // namespace thirdai::smx