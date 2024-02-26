#pragma once

#include <hashing/src/HashFunction.h>
#include <smx/src/autograd/Variable.h>
#include <smx/src/autograd/functions/LinearAlgebra.h>
#include <smx/src/modules/Module.h>
#include <smx/src/tensor/Init.h>
#include <memory>
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
    deregisterParameter("weight");
    registerParameter("weight", _weight);
  }

  const auto& bias() const { return _bias; }

  void setBias(VariablePtr b) {
    CHECK(b->tensor()->shape() == _bias->tensor()->shape(),
          "Shape must match in setBias.");
    CHECK(b->tensor()->dtype() == _bias->tensor()->dtype(),
          "Dtype must match in setBias.");

    _bias = std::move(b);
    deregisterParameter("bias");
    registerParameter("bias", _bias);
  }

 private:
  VariablePtr _weight;
  VariablePtr _bias;
};

class LshIndexConfig {
 public:
  LshIndexConfig(hashing::HashFunctionPtr hash_fn, size_t reservoir_size)
      : hash_fn(std::move(hash_fn)), reservoir_size(reservoir_size) {}

  hashing::HashFunctionPtr hash_fn;
  size_t reservoir_size;
};

class SparseLinear final : public Module {
 public:
  SparseLinear(size_t dim, size_t input_dim, float sparsity,
               const std::optional<LshIndexConfig>& lsh_index,
               size_t updates_per_rebuild = 4,
               size_t updates_per_new_hash_fn = 100)
      : _sparsity(sparsity) {
    _weight = Variable::make(
        smx::normal({dim, input_dim}, /*mean=*/0.0, /*stddev=*/0.01),
        /*requires_grad=*/true);

    _bias = Variable::make(smx::normal({dim}, /*mean=*/0.0, /*stddev=*/0.01),
                           /*requires_grad=*/true);

    registerParameter("weight", _weight);
    registerParameter("bias", _bias);

    if (lsh_index) {
      _neuron_index =
          LshIndex::make(lsh_index->hash_fn, lsh_index->reservoir_size,
                         dense(_weight->tensor()), updates_per_rebuild,
                         updates_per_new_hash_fn);
    } else {
      _neuron_index = LshIndex::autotune(
          dim, input_dim, sparsity, dense(_weight->tensor()),
          /* updates_per_rebuild=*/2, /*updates_per_new_hash_fn=*/11);
    }
  }

  VariablePtr forward(const VariablePtr& x,
                      const VariablePtr& labels = nullptr) {
    if (_sparsity < 1.0 && training()) {
      return linear(x, _weight, _bias, _sparsity, _neuron_index, labels);
    }
    return linear(x, _weight, _bias);
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
    deregisterParameter("weight");
    registerParameter("weight", _weight);
    _neuron_index->setWeight(dense(_weight->tensor()));
  }

  const auto& bias() const { return _bias; }

  void setBias(VariablePtr b) {
    CHECK(b->tensor()->shape() == _bias->tensor()->shape(),
          "Shape must match in setBias.");
    CHECK(b->tensor()->dtype() == _bias->tensor()->dtype(),
          "Dtype must match in setBias.");

    _bias = std::move(b);
    deregisterParameter("bias");
    registerParameter("bias", _bias);
  }

  std::function<void()> onUpdateCallback() {
    return [neuron_index = _neuron_index]() { neuron_index->onUpdate(); };
  }

  float sparsity() const { return _sparsity; }

  void setSparsity(float sparsity) { _sparsity = sparsity; }

  const auto& neuronIndex() const { return _neuron_index; }

  void autotuneHashTableRebuild(size_t n_batches, size_t batch_size) {
    if (auto lsh = std::dynamic_pointer_cast<LshIndex>(_neuron_index)) {
      lsh->autotuneHashTableRebuild(n_batches, batch_size);
    }
  }

 private:
  VariablePtr _weight;
  VariablePtr _bias;
  float _sparsity;
  NeuronIndexPtr _neuron_index;
};

}  // namespace thirdai::smx