#pragma once

#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <archive/src/Archive.h>
#include <cstddef>

namespace thirdai::bolt::tests {

class Noop final : public Op, public std::enable_shared_from_this<Noop> {
 private:
  explicit Noop(std::string name, uint32_t dim, uint32_t num_nonzeros)
      : Op(std::move(name)), _dim(dim), _num_nonzeros(num_nonzeros) {}

 public:
  static auto make(std::string name, uint32_t dim = 1,
                   uint32_t num_nonzeros = 1) {
    return std::shared_ptr<Noop>(new Noop(std::move(name), dim, num_nonzeros));
  }

  ComputationPtr apply(const ComputationList& inputs) {
    return Computation::make(shared_from_this(), inputs);
  }

  ComputationPtr applyToInputs(const ComputationList& inputs) final {
    return apply(inputs);
  }

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final {
    (void)with_optimizer;
    return nullptr;
  }

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final {
    (void)inputs;
    (void)output;
    (void)index_in_batch;
    (void)training;
  }

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final {
    (void)inputs;
    (void)output;
    (void)index_in_batch;
  }

  void updateParameters(float lr, uint32_t t) final {
    (void)lr;
    (void)t;
  }

  uint32_t dim() const final { return _dim; }

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final {
    (void)inputs;
    if (use_sparsity) {
      return _num_nonzeros;
    }
    return _dim;
  }

  void initOptimizer() final {}

  void disableSparseParameterUpdates() final {}

  void enableSparseParameterUpdates() final {}

  std::vector<std::vector<float>*> gradients() final { return {}; }

  std::vector<std::vector<float>*> parameters() final { return {}; }

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final {
    (void)inputs;
    (void)output;
    summary << "Noop";
  }

  void updateNumNonzeros(uint32_t new_num_nonzeros) {
    _num_nonzeros = new_num_nonzeros;
  }

 private:
  uint32_t _dim, _num_nonzeros;
};

class MockLoss final : public Loss {
 public:
  explicit MockLoss(ComputationList outputs_used)
      : _outputs_used(std::move(outputs_used)) {}

  static auto make(ComputationList outputs_used) {
    return std::make_shared<MockLoss>(std::move(outputs_used));
  }

  float loss(uint32_t i) const final {
    (void)i;
    return 0.0;
  }

  void gradients(uint32_t i, uint32_t batch_size) const final {
    (void)i;
    (void)batch_size;
  }

  ComputationList outputsUsed() const final { return _outputs_used; }

  ComputationList labels() const final { return {}; }

  ar::ConstArchivePtr toArchive() const final { return nullptr; }

 private:
  ComputationList _outputs_used;
};

inline ComputationPtr emptyInput() { return Input::make(/* dim= */ 1); }

}  // namespace thirdai::bolt::tests