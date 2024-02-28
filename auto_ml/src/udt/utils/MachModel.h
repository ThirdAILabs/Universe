#pragma once

#include <smx/src/autograd/Variable.h>
#include <smx/src/autograd/functions/Activations.h>
#include <smx/src/modules/Embedding.h>
#include <smx/src/modules/Linear.h>
#include <smx/src/modules/Module.h>

namespace thirdai::automl::udt {

class MachModel final : public smx::Module {
 public:
  MachModel(size_t input_dim, size_t hidden_dim, size_t output_dim,
            float output_sparsity) {
    emb = std::make_shared<smx::Embedding>(input_dim, hidden_dim);
    registerModule("emb", emb);

    out = std::make_shared<smx::SparseLinear>(output_dim, hidden_dim,
                                              output_sparsity);
    registerModule("output", out);
  }

  std::vector<smx::VariablePtr> forward(
      const std::vector<smx::VariablePtr>& inputs) final {
    if (inputs.size() == 2) {
      return {forward(inputs[0], inputs[1])};
    }
    if (inputs.size() == 1) {
      return {forward(inputs[0], nullptr)};
    }
    throw std::invalid_argument("MachModel can only take 1 or 2 arguments.");
  }

  smx::VariablePtr forward(const smx::VariablePtr& indices,
                           const smx::VariablePtr& labels = nullptr) const {
    auto embs = emb->forward(indices);
    embs = smx::relu(embs);
    return out->forward(embs, labels);
  }

  std::shared_ptr<smx::Embedding> emb;
  std::shared_ptr<smx::SparseLinear> out;
};

}  // namespace thirdai::automl::udt