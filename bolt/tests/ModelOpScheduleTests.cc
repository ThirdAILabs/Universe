#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>

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

class MockLoss : public loss::Loss {
 public:
  explicit MockLoss(std::vector<tensor::ActivationTensorPtr> outputs_used)
      : _outputs_used(std::move(outputs_used)) {}

  static auto make(std::vector<tensor::ActivationTensorPtr> outputs_used) {
    return std::make_shared<MockLoss>(std::move(outputs_used));
  }

  void gradients(uint32_t i, uint32_t batch_size) final {
    (void)i;
    (void)batch_size;
  }

  std::vector<tensor::ActivationTensorPtr> outputsUsed() const final {
    return _outputs_used;
  }

 private:
  std::vector<tensor::ActivationTensorPtr> _outputs_used;
};

TEST(OpSchedule, SingleOutput) {
  auto input = tensor::InputTensor::make(
      /* dim= */ 1, /* sparsity_type= */ tensor::SparsityType::Unknown,
      /* num_nonzeros= */ std::nullopt);

  auto act_1 = Noop::apply({input}, 1, "op_1")[0];
  auto act_2 = Noop::apply({input, act_1}, 1, "op_2")[0];
  auto act_3 = Noop::apply({input, act_1, act_2}, 1, "op_3")[0];
  auto act_4 = Noop::apply({input, act_1, act_2, act_3}, 1, "op_4")[0];
  auto act_5 = Noop::apply({input, act_1, act_2, act_3, act_4}, 1, "op_5")[0];

  auto loss = MockLoss::make({act_5});

  model::Model model(/* inputs= */ {input},
                     /* outputs= */ {act_5},
                     /* losses= */ {loss});

  ASSERT_EQ(model.ops().size(), 5);

  uint32_t cnt = 0;
  for (const auto& op : model.ops()) {
    ASSERT_EQ(op->name(), "op_" + std::to_string(++cnt));
  }
}

TEST(OpSchedule, MultipleOutputs) {
  auto input_1 = tensor::InputTensor::make(
      /* dim= */ 1, /* sparsity_type= */ tensor::SparsityType::Unknown,
      /* num_nonzeros= */ std::nullopt);

  auto input_2 = tensor::InputTensor::make(
      /* dim= */ 1, /* sparsity_type= */ tensor::SparsityType::Unknown,
      /* num_nonzeros= */ std::nullopt);

  auto input_3 = tensor::InputTensor::make(
      /* dim= */ 1, /* sparsity_type= */ tensor::SparsityType::Unknown,
      /* num_nonzeros= */ std::nullopt);

  auto act_1 = Noop::apply({input_1, input_2}, 1, "op_1")[0];
  auto act_2 = Noop::apply({input_2}, 1, "op_2")[0];
  auto act_3 = Noop::apply({input_3, act_1, act_2}, 2, "op_3");
  auto act_5 = Noop::apply({act_1, act_3[1]}, 1, "op_5")[0];
  auto act_4 = Noop::apply({input_3, act_3[0]}, 2, "op_4");
  auto act_6 = Noop::apply({act_3[0], act_3[1], act_4[1], act_5}, 1, "op_6")[0];
  auto act_7 = Noop::apply({act_3[1], act_5, act_6}, 1, "op_7")[0];

  auto loss = MockLoss::make({act_4[0], act_7});

  model::Model model(
      /* inputs= */ {input_1, input_2, input_3},
      /* outputs= */ {act_4[0], act_7},
      /* losses= */ {loss});

  ASSERT_EQ(model.ops().size(), 7);

  uint32_t cnt = 0;
  for (const auto& op : model.ops()) {
    ASSERT_EQ(op->name(), "op_" + std::to_string(++cnt));
  }
}

}  // namespace thirdai::bolt::nn::tests