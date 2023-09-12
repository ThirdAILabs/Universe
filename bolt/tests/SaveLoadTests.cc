#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <iterator>
#include <random>
#include <sstream>

namespace thirdai::bolt::nn::tests {

/**
 * These tests are to verify that different orderings of methods called on a
 * model do not produce errors from things like optimizer state being
 * uninitialized, or other missing data.
 */

class ModelInteractions {
 public:
  ModelInteractions() {
    auto input = Input::make(_input_dim);

    auto emb = Embedding::make(100, input->dim(), "relu")->apply(input);

    auto out = FullyConnected::make(_label_dim, emb->dim(), 0.2, "softmax")
                   ->apply(emb);

    auto loss = CategoricalCrossEntropy::make(out, Input::make(out->dim()));

    _model = Model::make({input}, {out}, {loss});
  }

  void train() {
    auto [x, y] = getInputs();
    _model->trainOnBatch(x, y);
    _model->updateParameters(0.001);
  }

  void predict() {
    auto [x, _] = getInputs();
    _model->forward(x);
  }

  void saveLoadWithoutOptimizer() {
    std::stringstream stream;
    _model->setSerializeOptimizer(false);
    _model->save_stream(stream);
    _model = Model::load_stream(stream);
  }

  void saveLoadWithOptimizer() {
    std::stringstream stream;
    _model->setSerializeOptimizer(true);
    _model->save_stream(stream);
    _model = Model::load_stream(stream);
  }

 private:
  std::pair<TensorList, TensorList> getInputs() const {
    std::vector<size_t> input_lengths = {42, 58, 37, 86};
    std::vector<size_t> label_lengths = {3, 7, 1, 6};

    auto input_indices = randomIndices(input_lengths, _input_dim);
    std::vector<float> input_values(input_indices.size(), 1.0);
    auto label_indices = randomIndices(label_lengths, _label_dim);
    std::vector<float> label_values(input_values.size(), 1.0);

    auto inputs =
        Tensor::sparse(std::move(input_indices), std::move(input_values),
                       std::move(input_lengths), _input_dim);

    auto labels =
        Tensor::sparse(std::move(label_indices), std::move(label_values),
                       std::move(label_lengths), _label_dim);

    return {{inputs}, {labels}};
  }

  static std::vector<uint32_t> randomIndices(const std::vector<size_t>& lengths,
                                             size_t dim) {
    std::vector<uint32_t> indices;

    std::mt19937 rng(dim);
    std::uniform_int_distribution<uint32_t> index_dist(0, dim - 1);
    for (uint32_t length : lengths) {
      std::generate_n(std::back_inserter(indices), length,
                      [&index_dist, &rng]() { return index_dist(rng); });
    }

    return indices;
  }

  size_t _input_dim = 500, _label_dim = 200;

  ModelPtr _model;
};

TEST(SaveLoadTests, TrainSaveLoadWithoutOptimizerTrain) {
  ModelInteractions model;

  model.train();
  model.saveLoadWithoutOptimizer();
  model.train();
  model.predict();
}

TEST(SaveLoadTests, TrainSaveLoadWithOptimizerTrain) {
  ModelInteractions model;

  model.train();
  model.saveLoadWithOptimizer();
  model.train();
  model.predict();
}

TEST(SaveLoadTests, TrainSaveLoadWithoutOptimizerPredict) {
  ModelInteractions model;

  model.train();
  model.saveLoadWithoutOptimizer();
  model.predict();
}

TEST(SaveLoadTests, TrainSaveLoadWithOptimizerPredict) {
  ModelInteractions model;

  model.train();
  model.saveLoadWithOptimizer();
  model.predict();
}

TEST(SaveLoadTests, NoTrainSaveLoadWithoutOptimizerTrain) {
  ModelInteractions model;

  model.saveLoadWithoutOptimizer();
  model.train();
  model.predict();
}

TEST(SaveLoadTests, NoTrainSaveLoadWithOptimizerTrain) {
  ModelInteractions model;

  model.saveLoadWithOptimizer();
  model.train();
  model.predict();
}

TEST(SaveLoadTests, NoTrainSaveLoadWithoutOptimizerPredict) {
  ModelInteractions model;

  model.saveLoadWithoutOptimizer();
  model.predict();
}

TEST(SaveLoadTests, NoTrainSaveLoadWithOptimizerPredict) {
  ModelInteractions model;

  model.saveLoadWithOptimizer();
  model.predict();
}

TEST(SaveLoadTests, PredictSaveLoadWithoutOptimizerTrain) {
  ModelInteractions model;

  model.predict();
  model.saveLoadWithoutOptimizer();
  model.train();
  model.predict();
}

TEST(SaveLoadTests, PredictSaveLoadWithOptimizerTrain) {
  ModelInteractions model;

  model.predict();
  model.saveLoadWithOptimizer();
  model.train();
  model.predict();
}

TEST(SaveLoadTests, PredictSaveLoadWithoutOptimizerPredict) {
  ModelInteractions model;

  model.predict();
  model.saveLoadWithoutOptimizer();
  model.predict();
}

TEST(SaveLoadTests, PredictSaveLoadWithOptimizerPredict) {
  ModelInteractions model;

  model.predict();
  model.saveLoadWithOptimizer();
  model.predict();
}

}  // namespace thirdai::bolt::nn::tests