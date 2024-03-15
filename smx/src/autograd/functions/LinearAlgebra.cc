#include "LinearAlgebra.h"
#include <bolt/src/utils/Timer.h>
#include <smx/src/autograd/Variable.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Tensor.h>
#include <utils/Logging.h>
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
  auto out =
      linear(x->tensor(), dense(w->tensor()), b ? dense(b->tensor()) : nullptr);

  GradFunc grad_func = [](const TensorPtr& out_grad,
                          const std::vector<VariablePtr>& inputs) {
    const auto& x = inputs.at(0);
    const auto& w = inputs.at(1);
    const auto& b = inputs.at(2);

    auto [x_grad, w_grad, b_grad] = linearGrad(
        x->tensor(), dense(w->tensor()), b ? dense(b->tensor()) : nullptr,
        dense(out_grad), x->requiresGrad());

    if (x->requiresGrad()) {
      x->addGradient(x_grad);
    }
    w->addGradient(w_grad);
    if (b) {
      b->addGradient(b_grad);
    }
  };

  return Variable::make(out, grad_func, {x, w, b});
}

VariablePtr linear(const VariablePtr& x, const VariablePtr& w,
                   const VariablePtr& b, float sparsity,
                   const NeuronIndexPtr& neuron_index,
                   const VariablePtr& labels) {
  auto out = linear(dense(x->tensor()), dense(w->tensor()),
                    b ? dense(b->tensor()) : nullptr, sparsity, neuron_index,
                    labels ? labels->tensor() : nullptr);

  GradFunc grad_func = [](const TensorPtr& out_grad,
                          const std::vector<VariablePtr>& inputs) {
    bolt::utils::Timer lin_back_timer;

    const auto& x = inputs.at(0);
    const auto& w = inputs.at(1);
    const auto& b = inputs.at(2);

    auto [x_grad, w_grad, b_grad] =
        linearGrad(dense(x->tensor()), dense(w->tensor()),
                   b ? dense(b->tensor()) : nullptr, csr(out_grad));

    x->addGradient(x_grad);
    w->addGradient(w_grad);
    if (b) {
      b->addGradient(b_grad);
    }

    lin_back_timer.stop();
    logging::info(fmt::format("smx linear backward | time {} ms",
                              lin_back_timer.milliseconds()));
  };

  return Variable::make(out, grad_func, {x, w, b});
}

}  // namespace thirdai::smx