#include "NN.h"
#include <bolt/src/utils/Timer.h>
#include <smx/src/autograd/Variable.h>
#include <smx/src/tensor/CsrTensor.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Tensor.h>
#include <utils/Logging.h>

namespace thirdai::smx {

VariablePtr embedding(const VariablePtr& indices, const VariablePtr& embs,
                      const VariablePtr& bias) {
  bolt::utils::Timer emb_for_timer;
  auto output = embedding(csr(indices->tensor()), dense(embs->tensor()),
                          bias ? dense(bias->tensor()) : nullptr);

  emb_for_timer.stop();
  logging::info(fmt::format("smx embedding forward | time {} ms",
                            emb_for_timer.milliseconds()));

  GradFunc grad_func = [](const TensorPtr& grad,
                          const std::vector<VariablePtr>& inputs) {
    bolt::utils::Timer emb_back_timer;

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

    emb_back_timer.stop();
    logging::info(fmt::format("smx embedding backward | time {} ms",
                              emb_back_timer.milliseconds()));
  };

  return Variable::make(output, grad_func, {indices, embs, bias});
}

}  // namespace thirdai::smx