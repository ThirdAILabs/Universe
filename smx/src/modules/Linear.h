#pragma once

#include <smx/src/autograd/functions/LinearAlgebra.h>
#include <smx/src/modules/Module.h>
#include <smx/src/tensor/Init.h>
#include <stdexcept>
#include <vector>

namespace thirdai::smx {

class Linear final : public UnaryModule {
 public:
  Linear(size_t dim, size_t input_dim) {
    _weight = Variable::make(
        smx::normal({dim, input_dim}, /*mean=*/0.0, /*stddev=*/0.01),
        /*requires_grad=*/true);

    _bias = Variable::make(smx::normal({dim}, /*mean=*/0.0, /*stddev=*/0.01),
                           /*requires_grad=*/true);
  }

  VariablePtr forward(const VariablePtr& x) final {
    return linear(x, _weight, _bias);
  }

  std::vector<VariablePtr> parameters() const final { return {_weight, _bias}; }

  const auto& weight() const { return _weight; }

  const auto& bias() const { return _bias; }

 private:
  VariablePtr _weight;
  VariablePtr _bias;
};

class SparseLinear final : public Module {
 public:
  SparseLinear(size_t dim, size_t input_dim, float sparsity,
               NeuronIndexPtr neuron_index)
      : _sparsity(sparsity) {
    _weight = Variable::make(
        smx::normal({dim, input_dim}, /*mean=*/0.0, /*stddev=*/0.01),
        /*requires_grad=*/true);

    _bias = Variable::make(smx::normal({dim}, /*mean=*/0.0, /*stddev=*/0.01),
                           /*requires_grad=*/true);

    if (neuron_index) {
      _neuron_index = std::move(neuron_index);
    } else {
      _neuron_index = LshIndex::autotune(
          dim, input_dim, sparsity, dense(_weight->tensor()),
          /* updates_per_rebuild=*/4, /*updates_per_new_hash_fn=*/100);
    }
  }

  std::vector<VariablePtr> forward(const std::vector<VariablePtr>& x) final {
    if (x.size() == 2) {
      return {linear(x[0], _weight, _bias, _sparsity, _neuron_index,
                     /*labels=*/x[1])};
    }
    if (x.size() == 1) {
      return {linear(x[0], _weight, _bias, _sparsity, _neuron_index,
                     /*labels=*/nullptr)};
    }
    throw std::invalid_argument(
        "Sparse linear can only take 1 or 2 arguments.");
  }

  std::vector<VariablePtr> parameters() const final { return {_weight, _bias}; }

  const auto& weight() const { return _weight; }

  const auto& bias() const { return _bias; }

  std::function<void()> onUpdateCallback() {
    return [neuron_index = _neuron_index]() { neuron_index->onUpdate(); };
  }

 private:
  VariablePtr _weight;
  VariablePtr _bias;
  float _sparsity;
  NeuronIndexPtr _neuron_index;
};

}  // namespace thirdai::smx