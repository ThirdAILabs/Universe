#include "NN.h"
#include <smx/src/autograd/Variable.h>
#include <smx/src/tensor/CsrTensor.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Tensor.h>

namespace thirdai::smx {

VariablePtr embedding(const VariablePtr& indices, const VariablePtr& embs,
                      const VariablePtr& bias) {
  auto output = embedding(csr(indices->tensor()), dense(embs->tensor()),
                          bias ? dense(bias->tensor()) : nullptr);

  GradFunc grad_func = [](const TensorPtr& grad,
                          const std::vector<VariablePtr>& inputs) {
    const auto& indices = csr(inputs.at(0)->tensor());
    const auto& embs = inputs.at(1);
    const auto& bias = inputs.at(2);

    if (embs->requiresGrad()) {
      auto [emb_grad, bias_grad] =
          embeddingGrad(indices, dense(grad), bias != nullptr);
      embs->addGradient(emb_grad);
      if (bias) {
        bias->addGradient(bias_grad);
      }
    }
  };

  return Variable::make(output, grad_func, {indices, embs, bias});
}

}  // namespace thirdai::smx