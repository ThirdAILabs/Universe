#include "NN.h"
#include <smx/src/autograd/Variable.h>
#include <smx/src/tensor/CsrTensor.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Tensor.h>

namespace thirdai::smx {

VariablePtr embedding(const VariablePtr& indices, const VariablePtr& embs,
                      bool reduce_mean) {
  auto output =
      embedding(csr(indices->tensor()), dense(embs->tensor()), reduce_mean);

  GradFunc grad_func = [reduce_mean](const TensorPtr& grad,
                                     const std::vector<VariablePtr>& inputs) {
    const auto& indices = csr(inputs.at(0)->tensor());
    const auto& embs = inputs.at(1);

    if (embs->requiresGrad()) {
      auto emb_grad = embeddingGrad(indices, dense(grad), reduce_mean);
      embs->addGradient(emb_grad);
    }
  };

  return Variable::make(output, grad_func, {indices, embs});
}

}  // namespace thirdai::smx