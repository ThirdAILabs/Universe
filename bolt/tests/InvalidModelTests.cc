#include "TestUtils.h"
#include "gtest/gtest.h"
#include <bolt/src/nn/model/Model.h>
#include <stdexcept>

namespace thirdai::bolt::nn::tests {

// NOLINTNEXTLINE (clang-tidy doesn't like #define)
#define CHECK_MODEL_EXCEPTION(statement, msg)                 \
  try {                                                       \
    statement;                                                \
    FAIL() << "Expected std::invalid_argument to be thrown."; \
  } catch (const std::invalid_argument& err) {                \
    ASSERT_EQ(err.what(), std::string(msg));                  \
  } catch (...) {                                             \
    FAIL() << "Expected std::invalid_argument to be thrown."; \
  }

TEST(InvalidModelTests, OutputInLossWithDependentOps) {
  auto input = emptyInput();

  auto act_2 = Noop::make("op_1")->apply({input});
  auto act_3 = Noop::make("op_2")->apply({act_2});

  auto loss = MockLoss::make({act_2, act_3});

  CHECK_MODEL_EXCEPTION(
      model::Model(/* inputs= */ {input}, /* outputs= */ {act_2, act_3},
                   /* losses= */ {loss}),
      "Outputs used in loss functions must not be inputs to any further ops. "
      "Found output 'tensor_2' with a dependent op.");
}

TEST(InvalidModelTests, AllOutputsUsedInLoss) {
  auto input = emptyInput();

  auto act_1 = Noop::make("op_1")->apply({input});
  auto act_2 = Noop::make("op_2")->apply({act_1});
  auto act_3 = Noop::make("op_3")->apply({act_1});

  auto loss = MockLoss::make({act_2});

  CHECK_MODEL_EXCEPTION(
      model::Model(/* inputs= */ {input}, /* outputs= */ {act_2, act_3},
                   /* losses= */ {loss}),
      "Specified output 'tensor_4' is not found in the computation graph "
      "created from traversing backward from the specified loss functions.");
}

TEST(InvalidModelTests, OutputsCannotBeReusedInLosses) {
  auto input = emptyInput();

  auto act_1 = Noop::make("op_1")->apply({input});
  auto act_2 = Noop::make("op_1")->apply({act_1});
  auto act_3 = Noop::make("op_1")->apply({act_1});

  auto loss_1 = MockLoss::make({act_2});
  auto loss_2 = MockLoss::make({act_2, act_3});

  CHECK_MODEL_EXCEPTION(
      model::Model(/* inputs= */ {input}, /* outputs= */ {act_2, act_3},
                   /* losses= */ {loss_1, loss_2}),
      "Only outputs can be used in losses and outputs cannot be reused in "
      "multiple losses. Found output 'tensor_3' which is either not an output "
      "or has already been used in a loss function.");
}

TEST(InvalidModelTests, UnusedInput) {
  auto input_1 = emptyInput();
  auto input_2 = emptyInput();

  auto act_1 = Noop::make("op_1")->apply({input_1});

  auto loss = MockLoss::make({act_1});

  CHECK_MODEL_EXCEPTION(
      model::Model(/* inputs= */ {input_1, input_2}, /* outputs= */ {act_1},
                   /* losses= */ {loss}),
      "Input 'tensor_2' was not used by any computation in the model.");
}

TEST(InvalidModelTests, MissingInput) {
  auto input_1 = emptyInput();
  auto input_2 = emptyInput();

  auto act_1 = Noop::make("op_1")->apply({input_1, input_2});

  auto loss = MockLoss::make({act_1});

  CHECK_MODEL_EXCEPTION(
      model::Model(/* inputs= */ {input_1}, /* outputs= */ {act_1},
                   /* losses= */ {loss}),
      "Model computation depends on input 'tensor_2' that is not present in "
      "the list of inputs to the model.");
}

}  // namespace thirdai::bolt::nn::tests