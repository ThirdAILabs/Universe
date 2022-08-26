#include <bolt_vector/tests/BoltVectorTestUtils.h>
#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/InMemoryDataset.h>
#include <algorithm>
#include <random>

namespace thirdai::dataset {

class SaveLoadInMemoryDatasetTestFixture : public testing::Test {
 public:
  SaveLoadInMemoryDatasetTestFixture()
      : _vector_length_dist(_min_vector_length, _max_vector_length),
        _active_neurons_dist(0, _max_index),
        _activations_gradients_dist(0.0, _max_activation_or_gradient),
        _rand(/* seed= */ 24092) {}

  void TearDown() override {
    ASSERT_FALSE(std::remove(_save_filename.c_str()));
  }

  BoltDatasetPtr generateRandomDataset(uint64_t n_batches,
                                       uint64_t batch_size) {
    std::vector<bolt::BoltBatch> batches;
    for (uint64_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
      std::vector<bolt::BoltVector> vectors;

      for (uint64_t vec_idx = 0; vec_idx < batch_size; vec_idx++) {
        uint64_t overall_index = batch_idx * batch_size + vec_idx;
        bool is_dense = (overall_index % 4) < 2;
        bool has_gradient = (overall_index % 2) == 0;

        vectors.push_back(
            generateVector(_vector_length_dist(_rand), is_dense, has_gradient));
      }

      batches.emplace_back(std::move(vectors));
    }

    return std::make_shared<BoltDataset>(std::move(batches));
  }

  std::string _save_filename = "./dataset_serialized";

 private:
  bolt::BoltVector generateVector(uint32_t len, bool is_dense,
                                  bool has_gradient) {
    bolt::BoltVector vector(len, is_dense, has_gradient);

    if (!is_dense) {
      std::generate(vector.active_neurons, vector.active_neurons + len,
                    [&]() { return _active_neurons_dist(_rand); });
    }

    std::generate(vector.activations, vector.activations + len,
                  [&]() { return _activations_gradients_dist(_rand); });

    if (has_gradient) {
      std::generate(vector.gradients, vector.gradients + len,
                    [&]() { return _activations_gradients_dist(_rand); });
    }

    return vector;
  }

  static constexpr uint32_t _min_vector_length = 10, _max_vector_length = 40;
  static constexpr uint32_t _max_index = 10000;
  static constexpr float _max_activation_or_gradient = 100.0;

  std::uniform_int_distribution<uint32_t> _vector_length_dist;
  std::uniform_int_distribution<uint32_t> _active_neurons_dist;
  std::uniform_real_distribution<float> _activations_gradients_dist;
  std::mt19937 _rand;
};

TEST_F(SaveLoadInMemoryDatasetTestFixture, SaveLoadBoltDataset) {
  auto dataset =
      generateRandomDataset(/* n_batches= */ 50, /* batch_size= */ 20);

  dataset->save(_save_filename);

  auto reloaded_dataset = BoltDataset::load(_save_filename);

  for (uint64_t batch_idx = 0; batch_idx < dataset->numBatches(); batch_idx++) {
    EXPECT_EQ(dataset->batchSize(batch_idx),
              reloaded_dataset->batchSize(batch_idx));
    for (uint64_t vec_idx = 0; vec_idx < dataset->batchSize(batch_idx);
         vec_idx++) {
      bolt::tests::BoltVectorTestUtils::assertBoltVectorsAreEqual(
          dataset->at(batch_idx)[vec_idx],
          reloaded_dataset->at(batch_idx)[vec_idx]);
    }
  }
}

}  // namespace thirdai::dataset