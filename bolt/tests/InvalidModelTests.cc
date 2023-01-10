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

TEST(InvalidModelTests, OutputWithDependentOps) {
  auto input = emptyInput();

  auto act_1 = Noop::make("act_1")->apply({input});
  auto act_2 = Noop::make("act_2")->apply({act_1});

  auto loss = MockLoss::make({act_1, act_2});

  // NOLINTNEXTLINE (clang-tidy doesn't like ASSERT_THROW)
  CHECK_MODEL_EXCEPTION(
      model::Model(/* inputs= */ {input}, /* outputs= */ {act_1, act_2},
                   /* losses= */ {loss}),
      "Outputs must not be inputs to any ops. Found output 'act_1' with a "
      "dependent op.");
}

TEST(InvalidModelTests, AllOutputsUsedInLoss) {
  auto input = emptyInput();

  auto act_1 = Noop::make("act_1")->apply({input});
  auto act_2 = Noop::make("act_2")->apply({act_1});
  auto act_3 = Noop::make("act_3")->apply({act_1});

  auto loss = MockLoss::make({act_2});

  // NOLINTNEXTLINE (clang-tidy doesn't like ASSERT_THROW)
  CHECK_MODEL_EXCEPTION(
      model::Model(/* inputs= */ {input}, /* outputs= */ {act_2, act_3},
                   /* losses= */ {loss}),
      "All outputs must be used by a loss. Found an output 'act_3' which is "
      "not used by any loss function.");
}

TEST(InvalidModelTests, OnlyOutputsUsedInLoss) {
  auto input = emptyInput();

  auto act_1 = Noop::make("act_1")->apply({input});
  auto act_2 = Noop::make("act_2")->apply({act_1});
  auto act_3 = Noop::make("act_3")->apply({act_1});

  auto loss_1 = MockLoss::make({act_1, act_2});
  auto loss_2 = MockLoss::make({act_3});

  // NOLINTNEXTLINE (clang-tidy doesn't like ASSERT_THROW)
  CHECK_MODEL_EXCEPTION(
      model::Model(/* inputs= */ {input}, /* outputs= */ {act_2, act_3},
                   /* losses= */ {loss_1, loss_2}),
      "Only outputs can be used in losses and outputs cannot be reused in "
      "multiple losses. Found tensor 'act_1' which is either not an output or "
      "or has already been used in a loss function.");
}

TEST(InvalidModelTests, OutputsCannotBeReusedInLosses) {
  auto input = emptyInput();

  auto act_1 = Noop::make("act_1")->apply({input});
  auto act_2 = Noop::make("act_1")->apply({act_1});
  auto act_3 = Noop::make("act_1")->apply({act_1});

  auto loss_1 = MockLoss::make({act_2});
  auto loss_2 = MockLoss::make({act_2, act_3});

  // NOLINTNEXTLINE (clang-tidy doesn't like ASSERT_THROW)
  CHECK_MODEL_EXCEPTION(
      model::Model(/* inputs= */ {input}, /* outputs= */ {act_2, act_3},
                   /* losses= */ {loss_1, loss_2}),
      "Only outputs can be used in losses and outputs cannot be reused in "
      "multiple losses. Found tensor 'act_2' which is either not an output or "
      "or has already been used in a loss function.");
}

}  // namespace thirdai::bolt::nn::tests