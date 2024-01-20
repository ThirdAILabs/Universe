#pragma once

#include <smx/src/autograd/Variable.h>
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

    registerParameter("weight", _weight);
    registerParameter("bias", _bias);
  }

  VariablePtr forward(const VariablePtr& x) final {
    return linear(x, _weight, _bias);
  }

  const auto& weight() const { return _weight; }

  void setWeight(VariablePtr w) {
    CHECK(w->tensor()->shape() == _weight->tensor()->shape(),
          "Shape must match in setWeight.");
    CHECK(w->tensor()->dtype() == _weight->tensor()->dtype(),
          "Dtype must match in setWeight.");
    _weight = std::move(w);
  }

  const auto& bias() const { return _bias; }

  void setBias(VariablePtr b) {
    CHECK(b->tensor()->shape() == _bias->tensor()->shape(),
          "Shape must match in setBias.");
    CHECK(b->tensor()->dtype() == _bias->tensor()->dtype(),
          "Dtype must match in setBias.");
    _bias = std::move(b);
  }

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

    registerParameter("weight", _weight);
    registerParameter("bias", _bias);

    if (neuron_index) {
      _neuron_index = std::move(neuron_index);
    } else {
      _neuron_index = LshIndex::autotune(
          dim, input_dim, sparsity, dense(_weight->tensor()),
          /* updates_per_rebuild=*/4, /*updates_per_new_hash_fn=*/100);
    }
  }

  VariablePtr forward(const VariablePtr& x,
                      const VariablePtr& labels = nullptr) {
    return linear(x, _weight, _bias, _sparsity, _neuron_index, labels);
  }

  std::vector<VariablePtr> forward(const std::vector<VariablePtr>& x) final {
    if (x.size() == 2) {
      return {forward(x[0], x[1])};
    }
    if (x.size() == 1) {
      return {forward(x[0])};
    }
    throw std::invalid_argument(
        "Sparse linear can only take 1 or 2 arguments.");
  }

  const auto& weight() const { return _weight; }

  void setWeight(VariablePtr w) {
    CHECK(w->tensor()->shape() == _weight->tensor()->shape(),
          "Shape must match in setWeight.");
    CHECK(w->tensor()->dtype() == _weight->tensor()->dtype(),
          "Dtype must match in setWeight.");
    _weight = std::move(w);
  }

  const auto& bias() const { return _bias; }

  void setBias(VariablePtr b) {
    CHECK(b->tensor()->shape() == _bias->tensor()->shape(),
          "Shape must match in setBias.");
    CHECK(b->tensor()->dtype() == _bias->tensor()->dtype(),
          "Dtype must match in setBias.");
    _bias = std::move(b);
  }

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