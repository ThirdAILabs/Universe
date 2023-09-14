#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Switch.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <gtest/gtest.h>
#include <memory>

namespace thirdai::bolt::nn::tests {

static uint32_t N_LAYERS = 5;

ModelPtr buildModel() {
  auto index = bolt::Input::make(N_LAYERS);
  auto input = bolt::Input::make(/* dim= */ 1);
  auto label = bolt::Input::make(/* dim= */ 1);
  auto output_op =
      bolt::Switch::make(N_LAYERS, /* dim= */ 1, /* input_dim= */ 1,
                         /* sparsity= */ 1.0,
                         /* activation= */ "linear");
  auto output = output_op->apply(index, input);

  for (uint32_t layer_id = 0; layer_id < N_LAYERS; layer_id++) {
    float bias = 0;
    float weight = layer_id;
    output_op->setWeights(layer_id, &weight);
    output_op->setBiases(layer_id, &bias);
  }

  auto loss = bolt::CategoricalCrossEntropy::make(output, label);
  return Model::make({index, input}, {output}, {loss});
}

TEST(SwitchOpTests, SwitchesToCorrectOp) {
  auto model = buildModel();

  auto index = Tensor::sparse(/* batch_size= */ N_LAYERS,
                              /* dim= */ N_LAYERS, /* nonzeros= */ 1);
  for (uint32_t vector_id = 0; vector_id < index->batchSize(); vector_id++) {
    index->getVector(vector_id).active_neurons[0] = vector_id;
    index->getVector(vector_id).activations[0] = 1.0;
  }

  auto input = Tensor::dense(/* batch_size= */ N_LAYERS, /* dim= */ 1);
  for (uint32_t vector_id = 0; vector_id < input->batchSize(); vector_id++) {
    input->getVector(vector_id).activations[0] = 1.0;
  }

  auto output =
      model->forward({index, input}, /* use_sparsity= */ false).front();
  for (uint32_t vector_id = 0; vector_id < output->batchSize(); vector_id++) {
    ASSERT_EQ(output->getVector(vector_id).activations[0],
              static_cast<float>(vector_id));
  }
}

}  // namespace thirdai::bolt::nn::tests