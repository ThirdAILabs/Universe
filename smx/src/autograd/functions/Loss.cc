#include "Loss.h"
#include <bolt/src/utils/Timer.h>
#include <smx/src/autograd/Variable.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Tensor.h>
#include <utils/Logging.h>
#include <stdexcept>

namespace thirdai::smx {

VariablePtr crossEntropy(const VariablePtr& logits, const VariablePtr& labels) {
  bolt::utils::Timer cce_for_timer;

  auto tmp = sparseCrossEntropy(logits->tensor(), labels->tensor());

  cce_for_timer.stop();
  logging::info(fmt::format("smx cce forward | time {} ms",
                            cce_for_timer.milliseconds()));

  auto loss = std::move(tmp.first);  // Lambda captures don't like auto[...]
  auto activations = std::move(tmp.second);

  GradFunc grad_func = [activations, labels](
                           const TensorPtr& grad,
                           const std::vector<VariablePtr>& inputs) {
    bolt::utils::Timer cce_back_timer;

    (void)grad;  // TODO(Nicholas) support weighted loss here

    auto logit_grad = sparseCrossEntropyGrad(activations, labels->tensor());

    inputs.at(0)->addGradient(logit_grad);

    cce_back_timer.stop();
    logging::info(fmt::format("smx cce backward | time {} ms",
                              cce_back_timer.milliseconds()));
  };

  return Variable::make(loss, grad_func, {logits});
}

VariablePtr binaryCrossEntropy(const VariablePtr& logits,
                               const VariablePtr& labels) {
  auto tmp = sparseBinaryCrossEntropy(logits->tensor(), labels->tensor());
  auto loss = std::move(tmp.first);  // Lambda captures don't like auto[...]
  auto activations = std::move(tmp.second);

  GradFunc grad_func = [activations, labels](
                           const TensorPtr& grad,
                           const std::vector<VariablePtr>& inputs) {
    (void)grad;  // TODO(Nicholas) support weighted loss here

    auto logit_grad =
        sparseBinaryCrossEntropyGrad(activations, labels->tensor());

    inputs.at(0)->addGradient(logit_grad);
  };

  return Variable::make(loss, grad_func, {logits});
}

}  // namespace thirdai::smx