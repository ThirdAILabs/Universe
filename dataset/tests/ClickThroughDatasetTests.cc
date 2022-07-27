#include <bolt/src/layers/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/DatasetLoaders.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

namespace thirdai::dataset {

const std::string _filename = "./clickthrough_dataset_test_file";
static const uint32_t _num_vectors = 11, _batch_size = 4;

class ClickThroughDatasetTestFixture : public ::testing::Test {
 public:
  ClickThroughDatasetTestFixture()
      : gen(590240),
        _label_dist(0, _num_classes - 1),
        _dense_feature_dist(0, _dense_feature_range),
        _categorical_feature_dist(0, _categorical_feature_range) {}

  struct TestClickThroughVec {
    uint32_t label;
    std::vector<uint32_t> dense_features;
    std::vector<uint32_t> categorical_features;
  };

  TestClickThroughVec createTestClickThroughVec() {
    TestClickThroughVec vec;

    vec.label = _label_dist(gen);

    for (uint32_t i = 0; i < _num_dense_features; i++) {
      uint32_t feature = _dense_feature_dist(gen);
      if ((feature % 10) == 0) {
        feature = 0;
      }
      vec.dense_features.push_back(feature);
    }

    std::unordered_set<uint32_t> categorical_features;
    while (categorical_features.size() < _num_categorical_features) {
      categorical_features.insert(_categorical_feature_dist(gen));
    }
    for (uint32_t c : categorical_features) {
      uint32_t feature = c;
      if ((feature % 10) == 0) {
        feature = 0;
      }
      vec.categorical_features.push_back(feature);
    }

    return vec;
  }

  void SetUp() override {
    for (uint32_t i = 0; i < _num_vectors; i++) {
      _ground_truths_vectors.push_back(createTestClickThroughVec());
    }

    std::ofstream output_file = dataset::SafeFileIO::ofstream(_filename);

    for (const auto& vec : _ground_truths_vectors) {
      output_file << vec.label;
      for (uint32_t d : vec.dense_features) {
        if (d == 0) {
          output_file << '\t';
        } else {
          output_file << '\t' << d;
        }
      }

      // If we want the categorical features in hexadecimal
      // output_file << std::hex;
      for (uint32_t c : vec.categorical_features) {
        if (c == 0) {
          output_file << '\t';
        } else {
          output_file << '\t' << c;
        }
      }

      output_file << std::endl;
    }

    output_file.close();
  }

  void TearDown() override { ASSERT_FALSE(std::remove(_filename.c_str())); }

  static uint32_t getNumDenseFeatures() { return _num_dense_features; }
  static uint32_t getNumCategoricalFeatures() {
    return _num_categorical_features;
  }

  std::vector<TestClickThroughVec> _ground_truths_vectors;

  std::mt19937 gen;
  std::uniform_int_distribution<uint32_t> _label_dist;
  std::uniform_int_distribution<uint32_t> _dense_feature_dist;
  std::uniform_int_distribution<uint32_t> _categorical_feature_dist;

  void verifySparseLabelBatch(const bolt::BoltBatch& labels,
                              uint32_t label_count_base) {
    ASSERT_TRUE(labels.getBatchSize() == _batch_size ||
                labels.getBatchSize() == _num_vectors % _batch_size);

    for (uint32_t v = 0; v < labels.getBatchSize(); v++) {
      ASSERT_EQ(labels[v].len, 1);
      ASSERT_EQ(labels[v].active_neurons[0],
                _ground_truths_vectors.at(label_count_base + v).label);
      ASSERT_EQ(labels[v].activations[0], 1.0);
    }
  }

  void verifyDenseInputBatch(const bolt::BoltBatch& dense_inputs,
                             uint32_t vec_count_base) {
    ASSERT_TRUE(dense_inputs.getBatchSize() == _batch_size ||
                dense_inputs.getBatchSize() == _num_vectors % _batch_size);

    for (uint32_t v = 0; v < dense_inputs.getBatchSize(); v++) {
      ASSERT_EQ(dense_inputs[v].len, getNumDenseFeatures());
      for (uint32_t i = 0; i < getNumDenseFeatures(); i++) {
        float val =
            _ground_truths_vectors.at(vec_count_base + v).dense_features.at(i);
        ASSERT_EQ(dense_inputs[v].activations[i], val);
      }
    }
  }

  void verifyTokenBatch(const BoltTokenBatch& tokens, uint32_t vec_count_base) {
    ASSERT_TRUE(tokens.getBatchSize() == _batch_size ||
                tokens.getBatchSize() == _num_vectors % _batch_size);

    for (uint32_t v = 0; v < tokens.getBatchSize(); v++) {
      ASSERT_EQ(tokens[v].size(), getNumCategoricalFeatures());
      for (uint32_t i = 0; i < tokens[v].size(); i++) {
        ASSERT_EQ(tokens[v].at(i), _ground_truths_vectors.at(vec_count_base + v)
                                       .categorical_features.at(i));
      }
    }
  }

 private:
  static const uint32_t _num_classes = 10, _num_dense_features = 7,
                        _num_categorical_features = 4,
                        _dense_feature_range = 100000,
                        _categorical_feature_range =
                            std::numeric_limits<uint32_t>::max();
};

TEST_F(ClickThroughDatasetTestFixture, InMemoryDatasetTestSparseLabel) {
  auto [dense_inputs, tokens, labels] = ClickThroughDatasetLoader::loadDataset(
      _filename, _batch_size, /* num_dense_features= */ getNumDenseFeatures(),
      /* max_num_categorical_features= */ getNumCategoricalFeatures(),
      /* delimiter= */ '\t');

  uint32_t label_count = 0;
  for (const auto& batch : *labels) {
    verifySparseLabelBatch(batch, label_count);
    label_count += batch.getBatchSize();
  }
  ASSERT_EQ(label_count, _num_vectors);

  uint32_t vec_count = 0;
  for (const auto& dense_input : *dense_inputs) {
    verifyDenseInputBatch(dense_input, vec_count);
    vec_count += dense_input.getBatchSize();
  }
  ASSERT_EQ(vec_count, _num_vectors);

  uint32_t token_count = 0;
  for (const auto& token_input : *tokens) {
    verifyTokenBatch(token_input, token_count);
    token_count += token_input.getBatchSize();
  }
  ASSERT_EQ(token_count, _num_vectors);
}

}  // namespace thirdai::dataset