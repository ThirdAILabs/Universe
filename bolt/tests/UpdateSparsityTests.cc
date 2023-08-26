#include "gtest/gtest.h"
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::bolt::tests {

constexpr uint32_t INPUT_DIM = 20, HIDDEN_DIM = 100, N_CLASSES = 10;

void testSparsityChanges(ModelPtr& model, const FullyConnectedPtr& fc,
                         const ComputationPtr& fc_output) {
  auto input_batch =
      Tensor::convert(BoltVector::singleElementSparseVector(0), INPUT_DIM);

  auto label_batch =
      Tensor::convert(BoltVector::singleElementSparseVector(0), N_CLASSES);

  model->trainOnBatch({input_batch}, {label_batch});

  EXPECT_EQ(fc_output->tensor()->nonzeros(), 50);

  fc->setSparsity(0.7, /* rebuild_hash_tables= */ true,
                  /* experimental_autotune= */ true);

  EXPECT_EQ(fc_output->tensor()->nonzeros(), 70);

  fc->setSparsity(0.3, /* rebuild_hash_tables= */ true,
                  /* experimental_autotune= */ true);

  EXPECT_EQ(fc_output->tensor()->nonzeros(), 30);
}

TEST(UpdateSparsityTests, ReallocateModelStateAfterSetSparsity) {
  auto input = Input::make(INPUT_DIM);

  auto fc = FullyConnected::make(
      /* dim= */ HIDDEN_DIM, /* input_dim= */ INPUT_DIM,
      /* sparsity= */ 0.5, /* activation= */ "relu");

  auto fc_output = fc->applyUnary(input);

  auto output = FullyConnected::make(/* dim= */ N_CLASSES,
                                     /* input_dim= */ HIDDEN_DIM,
                                     /* sparsity= */ 1.0,
                                     /* activation= */ "softmax")
                    ->applyUnary(fc_output);

  auto loss = CategoricalCrossEntropy::make(output, Input::make(N_CLASSES));

  auto model = Model::make({input}, {output}, {loss});

  std::string save_path = "./saved_sparse_model.tmp";
  model->save(save_path);

  testSparsityChanges(model, fc, fc_output);

  auto new_model = Model::load(save_path);

  testSparsityChanges(new_model,
                      FullyConnected::cast(new_model->opExecutionOrder().at(0)),
                      new_model->computationOrder().at(1));

  ASSERT_EQ(std::remove(save_path.c_str()), 0);
}

}  // namespace thirdai::bolt::tests