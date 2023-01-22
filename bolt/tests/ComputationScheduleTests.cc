#include "TestUtils.h"
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>

namespace thirdai::bolt::nn::tests {

TEST(ComputationScheduleTests, SingleOutput) {
  auto input = emptyInput();

  auto act_1 = Noop::make("op_1")->apply({input});
  auto act_2 = Noop::make("op_2")->apply({input, act_1});
  auto act_3 = Noop::make("op_3")->apply({input, act_1, act_2});
  auto act_4 = Noop::make("op_4")->apply({input, act_1, act_2, act_3});
  auto act_5 = Noop::make("op_5")->apply({input, act_1, act_2, act_3, act_4});

  auto loss = MockLoss::make({act_5});

  model::Model model(/* inputs= */ {input},
                     /* outputs= */ {act_5},
                     /* losses= */ {loss});

  ASSERT_EQ(model.opComputationOrder().size(), 5);
  uint32_t op_cnt = 0;
  for (const auto& op : model.opComputationOrder()) {
    ASSERT_EQ(op->name(), "op_" + std::to_string(++op_cnt));
  }

  ASSERT_EQ(model.computationOrder().size(), 6);
  uint32_t comp_cnt = 0;
  for (const auto& comp : model.computationOrder()) {
    ASSERT_EQ(comp->name(), "tensor_" + std::to_string(++comp_cnt));
  }
}

TEST(ComputationScheduleTests, MultipleOutputs) {
  auto input_1 = emptyInput();
  auto input_2 = emptyInput();
  auto input_3 = emptyInput();

  auto act_1 = Noop::make("op_1")->apply({input_1, input_2});
  auto act_2 = Noop::make("op_2")->apply({input_2, act_1});
  auto act_3 = Noop::make("op_3")->apply({input_3, act_1, act_2});
  auto act_4 = Noop::make("op_4")->apply({input_3, act_3});
  auto act_5 = Noop::make("op_5")->apply({act_1, act_3});
  auto act_6 = Noop::make("op_6")->apply({act_3, act_5});
  auto act_7 = Noop::make("op_7")->apply({act_3, act_5, act_6});

  auto loss = MockLoss::make({act_4, act_7});

  model::Model model(
      /* inputs= */ {input_1, input_2, input_3},
      /* outputs= */ {act_4, act_7},
      /* losses= */ {loss});

  ASSERT_EQ(model.opComputationOrder().size(), 7);

  std::vector<uint32_t> op_order_first_part = {1, 2, 3, 5, 6};
  for (uint32_t i = 0; i < 5; i++) {
    ASSERT_EQ(model.opComputationOrder()[i]->name(),
              "op_" + std::to_string(op_order_first_part[i]));
  }

  ASSERT_EQ(model.computationOrder().size(), 10);
  std::vector<uint32_t> comp_order_first_part = {1, 2, 3, 4, 5, 6, 8, 9};
  for (uint32_t i = 0; i < 8; i++) {
    ASSERT_EQ(model.computationOrder()[i]->name(),
              "tensor_" + std::to_string(comp_order_first_part[i]));
  }

  auto sixth_op = model.opComputationOrder()[5]->name();
  auto seventh_op = model.opComputationOrder()[6]->name();

  ASSERT_TRUE(sixth_op == "op_4" && seventh_op == "op_7" ||
              sixth_op == "op_7" && seventh_op == "op_4");

  auto ninth_comp = model.computationOrder()[8]->name();
  auto tenth_cmp = model.computationOrder()[9]->name();

  ASSERT_TRUE(ninth_comp == "tensor_7" && tenth_cmp == "tensor_10" ||
              ninth_comp == "tensor_10" && tenth_cmp == "tensor_7");
}

TEST(ComputationScheduleTests, Recurrence) {
  auto input_1 = emptyInput();
  auto input_2 = emptyInput();
  auto input_3 = emptyInput();
  auto input_4 = emptyInput();
  auto input_5 = emptyInput();

  auto op = Noop::make("recurrence");

  auto act_1 = op->apply({input_1, input_2});
  auto act_2 = op->apply({input_3, act_1});
  auto act_3 = op->apply({input_4, act_2});
  auto act_4 = op->apply({input_5, act_3});
  auto act_5 = Noop::make("output")->apply({act_4});

  auto loss = MockLoss::make({act_5});

  model::Model model(
      /* inputs= */ {input_1, input_2, input_3, input_4, input_5},
      /* outputs= */ {act_5}, /* losses= */ {loss});

  std::vector<std::string> op_order = {"recurrence", "recurrence", "recurrence",
                                       "recurrence", "output"};
  ASSERT_EQ(model.opComputationOrder().size(), 5);
  for (uint32_t i = 0; i < 5; i++) {
    ASSERT_EQ(model.opComputationOrder()[i]->name(), op_order[i]);
  }

  ASSERT_EQ(model.computationOrder().size(), 10);
  for (uint32_t i = 0; i < 10; i++) {
    ASSERT_EQ(model.computationOrder()[i]->name(),
              "tensor_" + std::to_string(i + 1));
  }
}

}  // namespace thirdai::bolt::nn::tests