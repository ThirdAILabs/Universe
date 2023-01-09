#pragma once

#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt::nn::tests {

class Noop final : public ops::Op, public std::enable_shared_from_this<Noop> {
 private:
  Noop(const std::vector<tensor::TensorPtr>& inputs, uint32_t n_outputs,
       std::string name)
      : ops::Op(std::move(name)) {
    for (const auto& input : inputs) {
      _inputs.push_back(input.get());
    }

    for (uint32_t i = 0; i < n_outputs; i++) {
      _outputs.push_back(tensor::ActivationTensor::make(
          /* dim= */ 1, /* sparse_nonzeros= */ 1, this));
    }
  }

 public:
  static auto apply(const std::vector<tensor::TensorPtr>& inputs,
                    uint32_t n_outputs, std::string name) {
    auto op =
        std::shared_ptr<Noop>(new Noop(inputs, n_outputs, std::move(name)));

    for (const auto& input : inputs) {
      input->addDependantOp(op);
    }

    return op->outputs();
  }

  void forward(uint32_t i, bool training) final {
    (void)i;
    (void)training;
  }

  void backpropagate(uint32_t i) final { (void)i; }

  void updateParameters(float lr, uint32_t t) final {
    (void)lr;
    (void)t;
  }

  std::vector<tensor::Tensor*> inputs() const final { return _inputs; }

  std::vector<tensor::ActivationTensorPtr> outputs() const final {
    return _outputs;
  }

  void disableSparseParameterUpdates() final {}

  void notifyInputSparsityChange() final {}

  void summary(std::ostream& summary) const final { summary << "Noop"; }

 private:
  std::vector<tensor::Tensor*> _inputs;
  std::vector<tensor::ActivationTensorPtr> _outputs;
};

class MockLoss final : public loss::Loss {
 public:
  explicit MockLoss(std::vector<tensor::ActivationTensorPtr> outputs_used)
      : _outputs_used(std::move(outputs_used)) {}

  static auto make(std::vector<tensor::ActivationTensorPtr> outputs_used) {
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

  std::vector<tensor::ActivationTensorPtr> outputsUsed() const final {
    return _outputs_used;
  }

 private:
  std::vector<tensor::ActivationTensorPtr> _outputs_used;
};

inline tensor::InputTensorPtr emptyInput() {
  return tensor::InputTensor::make(
      /* dim= */ 1, /* sparsity_type= */ tensor::SparsityType::Unknown,
      /* num_nonzeros= */ std::nullopt);
}

}  // namespace thirdai::bolt::nn::tests