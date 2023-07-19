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

  auto comp_2 = Noop::make("op_1")->apply({input});
  auto comp_3 = Noop::make("op_2")->apply({comp_2});

  auto loss = MockLoss::make({comp_2, comp_3});

  CHECK_MODEL_EXCEPTION(
      model::Model::make(/* inputs= */ {input}, /* outputs= */ {comp_2, comp_3},
                         /* losses= */ {loss}),
      "Computations used in loss functions must not be inputs to any further "
      "ops. Found computation that is used in a loss function and as an input "
      "to another computation.");
}

TEST(InvalidModelTests, AllOutputsUsedInLoss) {
  auto input = emptyInput();

  auto comp_1 = Noop::make("op_1")->apply({input});
  auto comp_2 = Noop::make("op_2")->apply({comp_1});
  auto comp_3 = Noop::make("op_3")->apply({comp_1});

  auto loss = MockLoss::make({comp_2});

  CHECK_MODEL_EXCEPTION(
      model::Model::make(/* inputs= */ {input}, /* outputs= */ {comp_2, comp_3},
                         /* losses= */ {loss}),
      "Model contains an output that is not found in the computation graph "
      "created from traversing backward from the specified loss functions.");
}

TEST(InvalidModelTests, OutputsCannotBeReusedInLosses) {
  auto input = emptyInput();

  auto comp_1 = Noop::make("op_1")->apply({input});
  auto comp_2 = Noop::make("op_1")->apply({comp_1});
  auto comp_3 = Noop::make("op_1")->apply({comp_1});

  auto loss_1 = MockLoss::make({comp_2});
  auto loss_2 = MockLoss::make({comp_2, comp_3});

  CHECK_MODEL_EXCEPTION(
      model::Model::make(/* inputs= */ {input}, /* outputs= */ {comp_2, comp_3},
                         /* losses= */ {loss_1, loss_2}),
      "Two loss functions cannot be applied to the same computation.");
}

TEST(InvalidModelTests, UnusedInput) {
  auto input_1 = emptyInput();
  auto input_2 = emptyInput();

  auto comp_1 = Noop::make("op_1")->apply({input_1});

  auto loss = MockLoss::make({comp_1});

  CHECK_MODEL_EXCEPTION(model::Model::make(/* inputs= */ {input_1, input_2},
                                           /* outputs= */ {comp_1},
                                           /* losses= */ {loss}),
                        "The input passed at index 1 was not used by any "
                        "computation in the model.");
}

TEST(InvalidModelTests, MissingInput) {
  auto input_1 = emptyInput();
  auto input_2 = emptyInput();

  auto comp_1 = Noop::make("op_1")->apply({input_1, input_2});

  auto loss = MockLoss::make({comp_1});

  CHECK_MODEL_EXCEPTION(
      model::Model::make(/* inputs= */ {input_1}, /* outputs= */ {comp_1},
                         /* losses= */ {loss}),
      "Model computation depends on an input that is not present in the list "
      "of inputs to the model.");
}

}  // namespace thirdai::bolt::nn::tests