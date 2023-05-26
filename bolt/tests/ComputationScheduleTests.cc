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

  auto comp_1 = Noop::make("op_1")->apply({input});
  auto comp_2 = Noop::make("op_2")->apply({input, comp_1});
  auto comp_3 = Noop::make("op_3")->apply({input, comp_1, comp_2});
  auto comp_4 = Noop::make("op_4")->apply({input, comp_1, comp_2, comp_3});
  auto comp_5 =
      Noop::make("op_5")->apply({input, comp_1, comp_2, comp_3, comp_4});

  auto loss = MockLoss::make({comp_5});

  auto model = model::Model::make(/* inputs= */ {input},
                                  /* outputs= */ {comp_5},
                                  /* losses= */ {loss});

  ASSERT_EQ(model->opExecutionOrder().size(), 5);
  uint32_t op_cnt = 0;
  for (const auto& op : model->opExecutionOrder()) {
    ASSERT_EQ(op->name(), "op_" + std::to_string(++op_cnt));
  }

  ASSERT_EQ(model->computationOrder().size(), 6);
  uint32_t comp_cnt = 0;
  for (const auto& comp : model->computationOrder()) {
    ASSERT_EQ(comp->name(), "tensor_" + std::to_string(++comp_cnt));
  }
}

TEST(ComputationScheduleTests, MultipleOutputs) {
  auto input_1 = emptyInput();
  auto input_2 = emptyInput();
  auto input_3 = emptyInput();

  auto comp_1 = Noop::make("op_1")->apply({input_1, input_2});
  auto comp_2 = Noop::make("op_2")->apply({input_2, comp_1});
  auto comp_3 = Noop::make("op_3")->apply({input_3, comp_1, comp_2});
  auto comp_4 = Noop::make("op_4")->apply({input_3, comp_3});
  auto comp_5 = Noop::make("op_5")->apply({comp_1, comp_3});
  auto comp_6 = Noop::make("op_6")->apply({comp_3, comp_5});
  auto comp_7 = Noop::make("op_7")->apply({comp_3, comp_5, comp_6});

  auto loss = MockLoss::make({comp_4, comp_7});

  auto model = model::Model::make(
      /* inputs= */ {input_1, input_2, input_3},
      /* outputs= */ {comp_4, comp_7},
      /* losses= */ {loss});

  ASSERT_EQ(model->opExecutionOrder().size(), 7);

  std::vector<uint32_t> op_order_first_part = {1, 2, 3, 5, 6};
  for (uint32_t i = 0; i < 5; i++) {
    ASSERT_EQ(model->opExecutionOrder()[i]->name(),
              "op_" + std::to_string(op_order_first_part[i]));
  }

  ASSERT_EQ(model->computationOrder().size(), 10);
  std::vector<uint32_t> comp_order_first_part = {1, 2, 3, 4, 5, 6, 8, 9};
  for (uint32_t i = 0; i < 8; i++) {
    ASSERT_EQ(model->computationOrder()[i]->name(),
              "tensor_" + std::to_string(comp_order_first_part[i]));
  }

  auto sixth_op = model->opExecutionOrder()[5]->name();
  auto seventh_op = model->opExecutionOrder()[6]->name();

  ASSERT_TRUE(sixth_op == "op_4" && seventh_op == "op_7" ||
              sixth_op == "op_7" && seventh_op == "op_4");

  auto ninth_comp = model->computationOrder()[8]->name();
  auto tenth_cmp = model->computationOrder()[9]->name();

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

  auto comp_1 = op->apply({input_1, input_2});
  auto comp_2 = op->apply({input_3, comp_1});
  auto comp_3 = op->apply({input_4, comp_2});
  auto comp_4 = op->apply({input_5, comp_3});
  auto comp_5 = Noop::make("output")->apply({comp_4});

  auto loss = MockLoss::make({comp_5});

  auto model = model::Model::make(
      /* inputs= */ {input_1, input_2, input_3, input_4, input_5},
      /* outputs= */ {comp_5}, /* losses= */ {loss});

  std::vector<std::string> op_order = {"recurrence", "recurrence", "recurrence",
                                       "recurrence", "output"};
  ASSERT_EQ(model->opExecutionOrder().size(), 5);
  for (uint32_t i = 0; i < 5; i++) {
    ASSERT_EQ(model->opExecutionOrder()[i]->name(), op_order[i]);
  }

  ASSERT_EQ(model->computationOrder().size(), 10);
  for (uint32_t i = 0; i < 10; i++) {
    ASSERT_EQ(model->computationOrder()[i]->name(),
              "tensor_" + std::to_string(i + 1));
  }
}

}  // namespace thirdai::bolt::nn::tests