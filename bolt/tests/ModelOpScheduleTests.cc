#include "TestUtils.h"
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>

namespace thirdai::bolt::nn::tests {

TEST(OpSchedule, SingleOutput) {
  auto input = emptyInput();

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
  auto input_1 = emptyInput();

  auto input_2 = emptyInput();

  auto input_3 = emptyInput();

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