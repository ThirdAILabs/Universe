#include "TestUtils.h"
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>

namespace thirdai::bolt::nn::tests {

TEST(ModelOpScheduleTests, SingleOutput) {
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

  ASSERT_EQ(model.ops().size(), 5);
  uint32_t op_cnt = 0;
  for (const auto& op : model.ops()) {
    ASSERT_EQ(op->name(), "op_" + std::to_string(++op_cnt));
  }

  ASSERT_EQ(model.tensors().size(), 5);
  uint32_t tensor_cnt = 0;
  for (const auto& tensor : model.tensors()) {
    ASSERT_EQ(tensor->name(), "act_" + std::to_string(++tensor_cnt));
  }
}

TEST(ModelOpScheduleTests, MultipleOutputs) {
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

  std::vector<uint32_t> first_part_of_order = {1, 2, 3, 5, 6};

  ASSERT_EQ(model.ops().size(), 7);
  ASSERT_EQ(model.tensors().size(), 7);

  for (uint32_t i = 0; i < 5; i++) {
    ASSERT_EQ(model.ops()[i]->name(),
              "op_" + std::to_string(first_part_of_order[i]));
    ASSERT_EQ(model.tensors()[i]->name(),
              "act_" + std::to_string(first_part_of_order[i]));
  }

  auto sixth_op = model.ops()[5]->name();
  auto seventh_op = model.ops()[6]->name();

  ASSERT_TRUE(sixth_op == "op_4" && seventh_op == "op_7" ||
              sixth_op == "op_7" && seventh_op == "op_4");

  auto sixth_tensor = model.tensors()[5]->name();
  auto seventh_tensor = model.tensors()[6]->name();

  ASSERT_TRUE(sixth_tensor == "act_4" && seventh_tensor == "act_7" ||
              sixth_tensor == "act_7" && seventh_tensor == "act_4");
}

TEST(ModelOpScheduleTests, TestRecurrence) {
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
  ASSERT_EQ(model.ops().size(), 5);
  for (uint32_t i = 0; i < 5; i++) {
    ASSERT_EQ(model.ops()[i]->name(), op_order[i]);
  }

  ASSERT_EQ(model.tensors().size(), 5);
  for (uint32_t i = 0; i < 5; i++) {
    ASSERT_EQ(model.tensors()[i]->name(), "act_" + std::to_string(i + 1));
  }
}

}  // namespace thirdai::bolt::nn::tests