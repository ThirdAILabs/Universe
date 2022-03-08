#include <gtest/gtest.h>
#include <dataset/src/Dataset.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

namespace thirdai::dataset {

const std::string _filename = "./svm_dataset_test_file";
static const uint32_t _num_vectors = 10000, _batch_size = 256;

class SvmDatasetTestFixture : public ::testing::Test {
 public:
  SvmDatasetTestFixture()
      : gen(590240),
        _label_dist(0, _num_classes - 1),
        _index_dist(0, _max_dim - 1),
        _value_dist(-_val_range, _val_range) {}

  struct TestSparseVec {
    std::vector<uint32_t> labels;
    std::vector<std::pair<uint32_t, float>> values;
  };

  TestSparseVec createTestSparseVec() {
    TestSparseVec vec;

    std::unordered_set<uint32_t> label_set;
    for (uint32_t i = 0; i < _num_labels; i++) {
      label_set.insert(_label_dist(gen));
    }
    vec.labels.insert(vec.labels.end(), label_set.begin(), label_set.end());

    std::unordered_set<uint32_t> index_set;
    for (uint32_t i = 0; i < _nonzeros; i++) {
      index_set.insert(_index_dist(gen));
    }
    for (uint32_t i : index_set) {
      float val = _value_dist(gen) * 0.125;
      vec.values.push_back({i, val});
    }

    return vec;
  }

  void SetUp() override {
    for (uint32_t i = 0; i < _num_vectors; i++) {
      _vectors.push_back(createTestSparseVec());
    }

    std::ofstream output_file(_filename);

    ASSERT_TRUE(output_file.is_open());
    ASSERT_TRUE(output_file.good());
    ASSERT_FALSE(output_file.bad());
    ASSERT_FALSE(output_file.fail());

    uint32_t v = 0;
    for (const auto& vec : _vectors) {
      output_file << vec.labels.at(0);
      for (uint32_t l = 1; l < vec.labels.size(); l++) {
        output_file << "," << vec.labels.at(l);
      }

      uint32_t i = 0;
      for (const auto& x : vec.values) {
        const char* space = " ";
        switch (i++ & 3) {
          case 0:
          case 1:
            break;
          case 2:
            space = "\t";
            break;
          case 3:
            space = " \t ";
        }
        output_file << space << x.first << ":" << x.second;
      }

      const char* end_of_line = "";
      switch (v++ & 3) {
        case 0:
          break;
        case 1:
          end_of_line = " ";
          break;
        case 2:
          end_of_line = "\t";
          break;
        case 3:
          end_of_line = " \t ";
      }

      output_file << end_of_line << std::endl;
    }

    output_file.close();
  }

  void TearDown() override { ASSERT_FALSE(std::remove(_filename.c_str())); }

  std::vector<TestSparseVec> _vectors;

  std::mt19937 gen;
  std::uniform_int_distribution<uint32_t> _label_dist;
  std::uniform_int_distribution<uint32_t> _index_dist;
  std::uniform_int_distribution<int32_t> _value_dist;

 private:
  static const uint32_t _num_labels = 10, _nonzeros = 100, _num_classes = 10000,
                        _max_dim = 100000;
  static const int32_t _val_range = 1000;
};

TEST_F(SvmDatasetTestFixture, InMemoryDatasetTest) {
  InMemoryDataset<SparseBatch> dataset(_filename, _batch_size,
                                       SvmSparseBatchFactory{});

  uint32_t vec_count = 0;
  for (const auto& batch : dataset) {
    ASSERT_TRUE(batch.getBatchSize() == _batch_size ||
                batch.getBatchSize() == _num_vectors % _batch_size);

    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      ASSERT_EQ(batch.id(v), vec_count);

      ASSERT_EQ(batch.labels(v).size(), _vectors.at(vec_count).labels.size());
      for (uint32_t i = 0; i < batch.labels(v).size(); i++) {
        ASSERT_EQ(batch.labels(v).at(i), _vectors.at(vec_count).labels.at(i));
      }

      ASSERT_EQ(batch[v].length(), _vectors[vec_count].values.size());
      for (uint32_t i = 0; i < batch[v].length(); i++) {
        ASSERT_EQ(batch.at(v)._indices[i],
                  _vectors.at(vec_count).values.at(i).first);
        ASSERT_EQ(batch.at(v)._values[i],
                  _vectors.at(vec_count).values.at(i).second);
      }

      vec_count++;
    }
  }
  ASSERT_EQ(vec_count, _num_vectors);
}

TEST_F(SvmDatasetTestFixture, StreamedDatasetTest) {
  StreamedDataset<SparseBatch> dataset(
      _filename, _batch_size, std::make_unique<SvmSparseBatchFactory>());

  uint32_t vec_count = 0;
  while (auto batch_opt = dataset.nextBatch()) {
    const SparseBatch& batch = *batch_opt;
    ASSERT_TRUE(batch.getBatchSize() == _batch_size ||
                batch.getBatchSize() == _num_vectors % _batch_size);

    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      ASSERT_EQ(batch.id(v), vec_count);

      ASSERT_EQ(batch.labels(v).size(), _vectors.at(vec_count).labels.size());
      for (uint32_t i = 0; i < batch.labels(v).size(); i++) {
        ASSERT_EQ(batch.labels(v).at(i), _vectors.at(vec_count).labels.at(i));
      }

      ASSERT_EQ(batch[v].length(), _vectors[vec_count].values.size());
      for (uint32_t i = 0; i < batch[v].length(); i++) {
        ASSERT_EQ(batch.at(v)._indices[i],
                  _vectors.at(vec_count).values.at(i).first);
        ASSERT_EQ(batch.at(v)._values[i],
                  _vectors.at(vec_count).values.at(i).second);
      }

      vec_count++;
    }
  }
  ASSERT_EQ(vec_count, _num_vectors);
}

TEST_F(SvmDatasetTestFixture, BoltSvmDatasetTest) {
  InMemoryDataset<BoltInputBatch> dataset(_filename, _batch_size,
                                          BoltSvmBatchFactory{});

  uint32_t vec_count = 0;
  for (const auto& batch : dataset) {
    ASSERT_TRUE(batch.getBatchSize() == _batch_size ||
                batch.getBatchSize() == _num_vectors % _batch_size);

    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      ASSERT_EQ(batch.labels(v).len, _vectors.at(vec_count).labels.size());
      for (uint32_t i = 0; i < batch.labels(v).len; i++) {
        ASSERT_EQ(batch.labels(v).active_neurons[i],
                  _vectors.at(vec_count).labels.at(i));
        ASSERT_FLOAT_EQ(batch.labels(v).activations[i],
                        1.0 / _vectors.at(vec_count).labels.size());
      }

      ASSERT_EQ(batch[v].len, _vectors[vec_count].values.size());
      for (uint32_t i = 0; i < batch[v].len; i++) {
        ASSERT_EQ(batch[v].active_neurons[i],
                  _vectors.at(vec_count).values.at(i).first);
        ASSERT_FLOAT_EQ(batch[v].activations[i],
                        _vectors.at(vec_count).values.at(i).second);
      }

      vec_count++;
    }
  }
  ASSERT_EQ(vec_count, _num_vectors);
}

}  // namespace thirdai::dataset