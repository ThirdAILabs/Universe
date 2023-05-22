#include "gtest/gtest.h"
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::bolt::nn::tests {

TEST(UpdateSparsityTests, ReallocateModelStateAfterSetSparsity) {
  constexpr uint32_t INPUT_DIM = 20, HIDDEN_DIM = 100, OUTPUT_DIM = 10;

  auto input = ops::Input::make(INPUT_DIM);

  auto fc = ops::FullyConnected::make(
      /* dim= */ HIDDEN_DIM, /* input_dim= */ INPUT_DIM,
      /* sparsity= */ 0.5, /* activation= */ "relu");

  auto fc_output = fc->apply(input);

  auto output = ops::FullyConnected::make(/* dim= */ OUTPUT_DIM,
                                          /* input_dim= */ HIDDEN_DIM,
                                          /* sparsity= */ 1.0,
                                          /* activation= */ "softmax")
                    ->apply(fc_output);

  auto loss =
      loss::CategoricalCrossEntropy::make(output, ops::Input::make(OUTPUT_DIM));

  auto model = model::Model::make({input}, {output}, {loss});

  auto input_batch = tensor::Tensor::convert(
      BoltVector::singleElementSparseVector(0), INPUT_DIM);

  auto label_batch = tensor::Tensor::convert(
      BoltVector::singleElementSparseVector(0), OUTPUT_DIM);

  model->trainOnBatch({input_batch}, {label_batch});

  EXPECT_EQ(fc_output->tensor()->nonzeros(), 50);

  fc->setSparsity(0.7, /* rebuild_hash_tables= */ true,
                  /* experimental_autotune= */ true);

  EXPECT_EQ(fc_output->tensor()->nonzeros(), 70);

  fc->setSparsity(0.3, /* rebuild_hash_tables= */ true,
                  /* experimental_autotune= */ true);

  EXPECT_EQ(fc_output->tensor()->nonzeros(), 30);
}

}  // namespace thirdai::bolt::nn::tests