#pragma once

#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt::nn::tests {

class Noop final : public ops::Op, public std::enable_shared_from_this<Noop> {
 private:
  explicit Noop(std::string name, uint32_t dim, uint32_t num_nonzeros)
      : ops::Op(std::move(name)), _dim(dim), _num_nonzeros(num_nonzeros) {}

 public:
  static auto make(std::string name, uint32_t dim = 1,
                   uint32_t num_nonzeros = 1) {
    return std::shared_ptr<Noop>(new Noop(std::move(name), dim, num_nonzeros));
  }

  tensor::ActivationTensorPtr apply(const tensor::TensorList& inputs) {
    return tensor::ActivationTensor::make(_dim, shared_from_this(), inputs);
  }

  void forward(const tensor::TensorList& inputs,
               tensor::ActivationTensor* output, uint32_t index_in_batch,
               bool training) final {
    (void)inputs;
    (void)output;
    (void)index_in_batch;
    (void)training;
  }

  void backpropagate(tensor::TensorList& inputs,
                     tensor::ActivationTensor* output,
                     uint32_t index_in_batch) final {
    (void)inputs;
    (void)output;
    (void)index_in_batch;
  }

  void updateParameters(float lr, uint32_t t) final {
    (void)lr;
    (void)t;
  }

  uint32_t numNonzerosInOutput(const tensor::TensorList& inputs,
                               bool use_sparsity) const final {
    (void)inputs;
    if (use_sparsity) {
      return _num_nonzeros;
    }
    return _dim;
  }

  void disableSparseParameterUpdates() final {}

  void summary(std::ostream& summary, const tensor::TensorList& inputs,
               const tensor::ActivationTensor* output) const final {
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