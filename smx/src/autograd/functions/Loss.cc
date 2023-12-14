#include "Loss.h"
#include <smx/src/autograd/Variable.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Tensor.h>
#include <stdexcept>

namespace thirdai::smx {

VariablePtr crossEntropy(const VariablePtr& logits, const TensorPtr& labels) {
  if (labels->dtype() == Dtype::u32) {
    CHECK(!labels->isSparse(), "Cannot have sparse u32 tensor for labels.");

    if (logits->tensor()->isSparse()) {
      throw std::invalid_argument(
          "Cross entropy on sparse tensor is not yet implemented.");
    }

    auto tmp = sparseCrossEntropy(dense(logits->tensor()), dense(labels));
    auto loss = std::move(tmp.first);  // Lambda captures don't like auto[...]
    auto activations = std::move(tmp.second);

    GradFunc grad_func = [activations, labels](
                             const TensorPtr& grad,
                             const std::vector<VariablePtr>& inputs) {
      (void)grad;  // TODO(Nicholas) support weighted loss here

      auto logit_grad = sparseCrossEntropyGrad(activations, dense(labels));

      inputs.at(0)->addGradient(logit_grad);
    };

    return Variable::make(loss, grad_func, {logits});
  }
  throw std::invalid_argument(
      "Dense cross entropy labels not yet implemented.");
}

}  // namespace thirdai::smx