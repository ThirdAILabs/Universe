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

  auto act_1 = Noop::apply({input}, /* n_outputs= */ 1, "act_1")[0];

  auto act_2 = Noop::apply({act_1}, /* n_outputs= */ 1, "act_2")[0];

  auto loss = MockLoss::make({act_1, act_2});

  // NOLINTNEXTLINE (clang-tidy doesn't like ASSERT_THROW)
  CHECK_MODEL_EXCEPTION(
      model::Model(/* inputs= */ {input}, /* outputs= */ {act_1, act_2},
                   /* losses= */ {loss}),
      "Outputs must not be inputs to any ops. Found output 'act_1' with a "
      "dependent op.");
}

TEST(InvalidModelTests, OnlyOutputsHaveNoDependents) {
  auto input = emptyInput();

  auto act_1 = Noop::apply({input}, /* n_outputs= */ 1, "act_1")[0];

  auto act_2 = Noop::apply({act_1}, /* n_outputs= */ 1, "act_2")[0];

  auto act_3 = Noop::apply({act_1}, /* n_outputs= */ 1, "act_3")[0];

  auto loss = MockLoss::make({act_2});

  // NOLINTNEXTLINE (clang-tidy doesn't like ASSERT_THROW)
  CHECK_MODEL_EXCEPTION(
      model::Model(/* inputs= */ {input}, /* outputs= */ {act_2},
                   /* losses= */ {loss}),
      "All non outputs must be used in at least one op. Found tensor 'act_3' "
      "that has no dependent ops and is not an output.");
}

TEST(InvalidModelTests, AllOutputsUsedInLoss) {
  auto input = emptyInput();

  auto act_1 = Noop::apply({input}, /* n_outputs= */ 1, "act_1")[0];

  auto act_2 = Noop::apply({act_1}, /* n_outputs= */ 1, "act_2")[0];

  auto act_3 = Noop::apply({act_1}, /* n_outputs= */ 1, "act_3")[0];

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

  auto act_1 = Noop::apply({input}, /* n_outputs= */ 1, "act_1")[0];

  auto act_2 = Noop::apply({act_1}, /* n_outputs= */ 1, "act_2")[0];

  auto act_3 = Noop::apply({act_1}, /* n_outputs= */ 1, "act_3")[0];

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

  auto act_1 = Noop::apply({input}, /* n_outputs= */ 1, "act_1")[0];

  auto act_2 = Noop::apply({act_1}, /* n_outputs= */ 1, "act_2")[0];

  auto act_3 = Noop::apply({act_1}, /* n_outputs= */ 1, "act_3")[0];

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