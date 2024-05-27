#include "TestUtils.h"
#include "gtest/gtest.h"
#include <bolt/src/nn/model/Model.h>
#include <stdexcept>

namespace thirdai::bolt::tests {

// NOLINTNEXTLINE (clang-tidy doesn't like #define)
#define CHECK_MODEL_EXCEPTION(statement, msg, exception_type)       \
  try {                                                             \
    statement;                                                      \
    FAIL() << "Expected exception to be thrown.";                   \
  } catch (const exception_type& err) {                             \
    ASSERT_EQ(err.what(), std::string(msg));                        \
  } catch (...) {                                                   \
    FAIL() << "Expected different type of exception to be thrown."; \
  }

TEST(InvalidModelTests, OutputInLossWithDependentOps) {
  auto input = emptyInput();

  auto comp_2 = Noop::make("op_1")->apply({input});
  auto comp_3 = Noop::make("op_2")->apply({comp_2});

  auto loss = MockLoss::make({comp_2, comp_3});

  CHECK_MODEL_EXCEPTION(
      Model::make(/* inputs= */ {input}, /* outputs= */ {comp_2, comp_3},
                  /* losses= */ {loss}),
      "Computations used in loss functions must not be inputs to any further "
      "ops. Found computation that is used in a loss function and as an input "
      "to another computation.",
      std::invalid_argument);
}

TEST(InvalidModelTests, AllOutputsUsedInLoss) {
  auto input = emptyInput();

  auto comp_1 = Noop::make("op_1")->apply({input});
  auto comp_2 = Noop::make("op_2")->apply({comp_1});
  auto comp_3 = Noop::make("op_3")->apply({comp_1});

  auto loss = MockLoss::make({comp_2});

  CHECK_MODEL_EXCEPTION(
      Model::make(/* inputs= */ {input}, /* outputs= */ {comp_2, comp_3},
                  /* losses= */ {loss}),
      "Model contains an output that is not found in the computation graph "
      "created from traversing backward from the specified loss functions.",
      std::invalid_argument);
}

TEST(InvalidModelTests, OutputsCannotBeReusedInLosses) {
  auto input = emptyInput();

  auto comp_1 = Noop::make("op_1")->apply({input});
  auto comp_2 = Noop::make("op_1")->apply({comp_1});
  auto comp_3 = Noop::make("op_1")->apply({comp_1});

  auto loss_1 = MockLoss::make({comp_2});
  auto loss_2 = MockLoss::make({comp_2, comp_3});

  CHECK_MODEL_EXCEPTION(
      Model::make(/* inputs= */ {input}, /* outputs= */ {comp_2, comp_3},
                  /* losses= */ {loss_1, loss_2}),
      "Two loss functions cannot be applied to the same computation.",
      std::invalid_argument);
}

TEST(InvalidModelTests, UnusedInput) {
  auto input_1 = emptyInput();
  auto input_2 = emptyInput();

  auto comp_1 = Noop::make("op_1")->apply({input_1});

  auto loss = MockLoss::make({comp_1});

  CHECK_MODEL_EXCEPTION(Model::make(/* inputs= */ {input_1, input_2},
                                    /* outputs= */ {comp_1},
                                    /* losses= */ {loss}),
                        "The input passed at index 1 was not used by any "
                        "computation in the model.",
                        std::invalid_argument);
}

TEST(InvalidModelTests, MissingInput) {
  auto input_1 = emptyInput();
  auto input_2 = emptyInput();

  auto comp_1 = Noop::make("op_1")->apply({input_1, input_2});

  auto loss = MockLoss::make({comp_1});

  CHECK_MODEL_EXCEPTION(
      Model::make(/* inputs= */ {input_1}, /* outputs= */ {comp_1},
                  /* losses= */ {loss}),
      "Model computation depends on an input that is not present in the list "
      "of inputs to the model.",
      std::invalid_argument);
}

TEST(InvalidModelTests, DuplidateOpNames) {
  auto input = emptyInput();

  auto comp_1 = Noop::make("op_1")->apply({input});
  auto comp_2 = Noop::make("op_1")->apply({comp_1});

  auto loss = MockLoss::make({comp_2});

  CHECK_MODEL_EXCEPTION(
      Model::make(/* inputs= */ {input}, /* outputs= */ {comp_2},
                  /* losses= */ {loss}),
      "Found multiple Ops in model with the name 'op_1'. All ops in a model "
      "must have unique names. The name of the op can be updated with `op.name "
      "= 'new_name'`.",
      std::invalid_argument);
}

TEST(InvalidModelTests, ComputationsCannotBeReused) {
  auto input_a = emptyInput();
  auto comp_a_1 = Noop::make("op_1")->apply({input_a});
  auto comp_a_2 = Noop::make("op_2")->apply({comp_a_1});
  auto loss_a = MockLoss::make({comp_a_2});

  auto model_a = Model::make(/* inputs= */ {input_a}, /* outputs= */ {comp_a_2},
                             /* losses= */ {loss_a});

  auto comp_b = Noop::make("op_3")->apply({comp_a_1});
  auto loss_b = MockLoss::make({comp_b});

  CHECK_MODEL_EXCEPTION(
      Model::make(/* inputs= */ {input_a}, /* outputs= */ {comp_b},
                  /* losses= */ {loss_b}),
      "Computations should only be named by the model, and computations should "
      "not be reused between models.",
      std::runtime_error);
}

}  // namespace thirdai::bolt::tests