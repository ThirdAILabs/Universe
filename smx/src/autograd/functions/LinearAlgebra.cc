#include "LinearAlgebra.h"
#include <smx/src/autograd/Variable.h>
#include <smx/src/tensor/Functions.h>
#include <vector>

namespace thirdai::smx {

VariablePtr add(const VariablePtr& a, const VariablePtr& b) {
  auto out = add(a->tensor(), b->tensor());

  GradFunc grad_func = [](const TensorPtr& out_grad,
                          const std::vector<VariablePtr>& inputs) {
    if (inputs.at(0)->requiresGrad()) {
      inputs.at(0)->addGradient(out_grad);
    }

    if (inputs.at(1)->requiresGrad()) {
      inputs.at(1)->addGradient(out_grad);
    }
  };

  return Variable::make(out, grad_func, {a, b});
}

VariablePtr linear(const VariablePtr& x, const VariablePtr& w,
                   const VariablePtr& b) {
  auto out = linear(dense(x->tensor()), dense(w->tensor()), dense(b->tensor()));

  GradFunc grad_func = [](const TensorPtr& out_grad,
                          const std::vector<VariablePtr>& inputs) {
    const auto& x = inputs.at(0);
    const auto& w = inputs.at(1);
    const auto& b = inputs.at(2);

    auto [x_grad, w_grad, b_grad] =
        linearGrad(dense(x->tensor()), dense(w->tensor()), dense(b->tensor()),
                   out_grad, x->requiresGrad());

    if (x->requiresGrad()) {
      x->addGradient(x_grad);
    }
    w->addGradient(w_grad);
    b->addGradient(b_grad);
  };

  return Variable::make(out, grad_func, {x, w, b});
}

}  // namespace thirdai::smx