#pragma once

#include <smx/src/autograd/functions/NN.h>
#include <smx/src/modules/Module.h>
#include <smx/src/tensor/Init.h>
#include <vector>

namespace thirdai::smx {

class Embedding final : public UnaryModule {
 public:
  Embedding(size_t n_embs, size_t emb_dim, bool reduce_mean = true)
      : _reduce_mean(reduce_mean) {
    _embs = Variable::make(
        smx::normal({n_embs, emb_dim}, /*mean=*/0.0, /*stddev=*/0.01),
        /*requires_grad=*/true);
  }

  VariablePtr forward(const VariablePtr& indices) final {
    return embedding(indices, _embs, _reduce_mean);
  }

  std::vector<VariablePtr> parameters() const final { return {_embs}; }

 private:
  VariablePtr _embs;
  bool _reduce_mean;
};

}  // namespace thirdai::smx