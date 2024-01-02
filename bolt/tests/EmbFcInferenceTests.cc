#include <bolt/src/inference/EmbFcInference.h>
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Input.h>
#include <gtest/gtest.h>

namespace thirdai::bolt {

TEST(EmbFcInferenceTests, OutputsMatch) {
  size_t input_dim = 100, emb_dim = 50, fc_dim = 80;

  auto input = Input::make(input_dim);
  auto emb = Embedding::make(emb_dim, input_dim, "relu");
  auto hidden = emb->apply(input);
  auto fc = FullyConnected::make(fc_dim, emb_dim, 0.1, "sigmoid");
  auto output = fc->apply(hidden);

  auto loss = BinaryCrossEntropy::make(output, Input::make(fc_dim));

  auto model = Model::make({input}, {output}, {loss});

  std::vector<uint32_t> input_indices = {1,  49, 2,  4,  90, 65, 24, 50, 32,
                                         62, 88, 79, 48, 17, 13, 8,  34};
  std::vector<float> input_values(input_indices.size(), 1.0);
  std::vector<size_t> lens = {5, 3, 4, 5};

  auto tensor =
      Tensor::sparse(std::move(input_indices), std::move(input_values),
                     std::move(lens), input_dim);

  EmbFcInference inf(emb, fc);

  auto model_output = model->forward({tensor}).at(0);
  auto inf_output = inf.forward(tensor);

  ASSERT_EQ(model_output->batchSize(), 4);
  ASSERT_EQ(inf_output->batchSize(), 4);

  for (size_t i = 0; i < 4; i++) {
    ASSERT_TRUE(model_output->getVector(i).isDense());
    ASSERT_TRUE(inf_output->getVector(i).isDense());

    ASSERT_EQ(model_output->getVector(i).len, fc_dim);
    ASSERT_EQ(inf_output->getVector(i).len, fc_dim);

    for (size_t j = 0; j < fc_dim; j++) {
      ASSERT_FLOAT_EQ(model_output->getVector(i).activations[j],
                      inf_output->getVector(i).activations[j]);
    }
  }
}

}  // namespace thirdai::bolt