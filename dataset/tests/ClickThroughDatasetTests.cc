#include <bolt/src/layers/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/Datasets.h>
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
      if (feature % 10 == 0) {
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
      if (feature % 10 == 0) {
        feature = 0;
      }
      vec.categorical_features.push_back(feature);
    }

    return vec;
  }

  void SetUp() override {
    for (uint32_t i = 0; i < _num_vectors; i++) {
      _vectors.push_back(createTestClickThroughVec());
    }

    std::ofstream output_file = dataset::SafeFileIO::ofstream(_filename);

    for (const auto& vec : _vectors) {
      output_file << std::dec << vec.label;
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

  std::vector<TestClickThroughVec> _vectors;

  std::mt19937 gen;
  std::uniform_int_distribution<uint32_t> _label_dist;
  std::uniform_int_distribution<uint32_t> _dense_feature_dist;
  std::uniform_int_distribution<uint32_t> _categorical_feature_dist;

  void verifyDataBatch(const ClickThroughBatch& batch,
                       uint32_t vec_count_base) {
    ASSERT_TRUE(batch.getBatchSize() == _batch_size ||
                batch.getBatchSize() == _num_vectors % _batch_size);

    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      // CHeck dense features are correct.
      ASSERT_EQ(batch[v].len, getNumDenseFeatures());
      for (uint32_t i = 0; i < getNumDenseFeatures(); i++) {
        float val = _vectors.at(vec_count_base + v).dense_features.at(i);
        ASSERT_EQ(batch.at(v).activations[i], val);
      }

      // Check Categorical features are correct.
      ASSERT_EQ(batch.categoricalFeatures(v).size(),
                getNumCategoricalFeatures());
      for (uint32_t i = 0; i < getNumCategoricalFeatures(); i++) {
        ASSERT_EQ(batch.categoricalFeatures(v).at(i),
                  _vectors.at(vec_count_base + v).categorical_features.at(i));
      }
    }
  }

  void verifyLabelBatch(const bolt::BoltBatch& labels,
                        uint32_t label_count_base, bool sparse_labels) {
    ASSERT_TRUE(labels.getBatchSize() == _batch_size ||
                labels.getBatchSize() == _num_vectors % _batch_size);

    for (uint32_t v = 0; v < labels.getBatchSize(); v++) {
      // Check labels are correct.
      ASSERT_EQ(labels[v].len, 1);
      if (sparse_labels) {
        ASSERT_EQ(labels[v].active_neurons[0],
                  _vectors.at(label_count_base + v).label);
        ASSERT_EQ(labels[v].activations[0], 1.0);
      } else {
        ASSERT_EQ(labels[v].active_neurons, nullptr);
        ASSERT_EQ(labels[v].activations[0],
                  _vectors.at(label_count_base + v).label);
      }
    }
  }

  void runClickThroughDatasetTest(bool sparse_labels) {
    auto dataset = ClickThroughDatasetLoader::loadDataset(
        _filename, _batch_size, getNumDenseFeatures(),
        getNumCategoricalFeatures(), sparse_labels);

    uint32_t vec_count = 0;
    for (const auto& batch : *dataset.data) {
      verifyDataBatch(batch, vec_count);
      vec_count += batch.getBatchSize();
    }
    ASSERT_EQ(vec_count, _num_vectors);

    uint32_t label_count = 0;
    for (const auto& batch : *dataset.labels) {
      verifyLabelBatch(batch, label_count, sparse_labels);
      label_count += batch.getBatchSize();
    }
    ASSERT_EQ(label_count, _num_vectors);
  }

 private:
  static const uint32_t _num_classes = 10, _num_dense_features = 7,
                        _num_categorical_features = 4,
                        _dense_feature_range = 100000,
                        _categorical_feature_range =
                            std::numeric_limits<uint32_t>::max();
};

TEST_F(ClickThroughDatasetTestFixture, InMemoryDatasetTestSparseLabel) {
  runClickThroughDatasetTest(/* sparse_labels= */ true);
}

TEST_F(ClickThroughDatasetTestFixture, InMemoryDatasetTestDenseLabel) {
  runClickThroughDatasetTest(/* sparse_labels= */ false);
}

}  // namespace thirdai::dataset