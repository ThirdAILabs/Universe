#include <gtest/gtest.h>
#include <dataset/src/Dataset.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

namespace thirdai::utils::dataset_tests {

const std::string _filename = "./clickthrough_dataset_test_file";
static const uint32_t _num_vectors = 11, _batch_size = 4;

class CLickThroughDatasetTestFixture : public ::testing::Test {
 public:
  CLickThroughDatasetTestFixture()
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

    std::ofstream output_file(_filename);

    ASSERT_TRUE(output_file.is_open());
    ASSERT_TRUE(output_file.good());
    ASSERT_FALSE(output_file.bad());
    ASSERT_FALSE(output_file.fail());

    for (const auto& vec : _vectors) {
      output_file << std::dec << vec.label;
      for (uint32_t d : vec.dense_features) {
        if (d == 0) {
          output_file << '\t';
        } else {
          output_file << '\t' << d;
        }
      }

      output_file << std::hex;
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

  // void TearDown() override { ASSERT_FALSE(std::remove(_filename.c_str())); }

  static uint32_t getNumDenseFeatures() { return _num_dense_features; }
  static uint32_t getNumCategoricalFeatures() {
    return _num_categorical_features;
  }

  std::vector<TestClickThroughVec> _vectors;

  std::mt19937 gen;
  std::uniform_int_distribution<uint32_t> _label_dist;
  std::uniform_int_distribution<uint32_t> _dense_feature_dist;
  std::uniform_int_distribution<uint32_t> _categorical_feature_dist;

  void verifyBatch(const ClickThroughBatch& batch, uint32_t vec_count_base) {
    ASSERT_TRUE(batch.getBatchSize() == _batch_size ||
                batch.getBatchSize() == _num_vectors % _batch_size);

    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      ASSERT_EQ(batch.id(v), vec_count_base + v);

      ASSERT_EQ(batch.label(v), _vectors.at(vec_count_base + v).label);

      ASSERT_EQ(batch[v].dim(), getNumDenseFeatures());
      for (uint32_t i = 0; i < getNumDenseFeatures(); i++) {
        float val = _vectors.at(vec_count_base + v).dense_features.at(i);
        ASSERT_EQ(batch.at(v)._values[i], val);
      }

      ASSERT_EQ(batch.categoricalFeatures(v).size(),
                getNumCategoricalFeatures());
      for (uint32_t i = 0; i < getNumCategoricalFeatures(); i++) {
        ASSERT_EQ(batch.categoricalFeatures(v).at(i),
                  _vectors.at(vec_count_base + v).categorical_features.at(i));
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

TEST_F(CLickThroughDatasetTestFixture, InMemoryDatasetTest) {
  InMemoryDataset<ClickThroughBatch> dataset(
      _filename, _batch_size,
      ClickThroughBatchFactory(getNumDenseFeatures(),
                               getNumCategoricalFeatures()));

  uint32_t vec_count = 0;
  for (const auto& batch : dataset) {
    verifyBatch(batch, vec_count);
    vec_count += batch.getBatchSize();
  }
  ASSERT_EQ(vec_count, _num_vectors);
}

TEST_F(CLickThroughDatasetTestFixture, StreamedDatasetTest) {
  StreamedDataset<ClickThroughBatch> dataset(
      _filename, _batch_size,
      std::make_unique<ClickThroughBatchFactory>(getNumDenseFeatures(),
                                                 getNumCategoricalFeatures()));

  uint32_t vec_count = 0;
  while (auto batch_opt = dataset.nextBatch()) {
    verifyBatch(*batch_opt, vec_count);
    vec_count += batch_opt->getBatchSize();
  }
  ASSERT_EQ(vec_count, _num_vectors);
}

}  // namespace thirdai::utils::dataset_tests