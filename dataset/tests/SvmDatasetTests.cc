#include <gtest/gtest.h>
#include <dataset/src/DatasetLoaders.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

namespace thirdai::dataset {

const std::string filename = "./svm_dataset_test_file";
static constexpr uint32_t num_vectors = 10000;
static constexpr uint32_t batch_size = 256;
static constexpr uint32_t num_labels = 10, nonzeros = 100, num_classes = 10000,
                          max_dim = 100000;
static constexpr int32_t val_range = 1000;

class SvmDatasetTestFixture : public ::testing::Test {
 public:
  SvmDatasetTestFixture()
      : gen(590240),
        _label_dist(0, num_classes - 1),
        _index_dist(0, max_dim - 1),
        _value_dist(-val_range, val_range) {}

  struct TestSparseVec {
    std::vector<uint32_t> labels;
    std::vector<std::pair<uint32_t, float>> values;
  };

  TestSparseVec createTestSparseVec() {
    TestSparseVec vec;

    std::unordered_set<uint32_t> label_set;
    for (uint32_t i = 0; i < num_labels; i++) {
      label_set.insert(_label_dist(gen));
    }
    vec.labels.insert(vec.labels.end(), label_set.begin(), label_set.end());

    std::unordered_set<uint32_t> index_set;
    for (uint32_t i = 0; i < nonzeros; i++) {
      index_set.insert(_index_dist(gen));
    }
    for (uint32_t i : index_set) {
      float val = _value_dist(gen) * 0.125;
      vec.values.push_back({i, val});
    }

    return vec;
  }

  void SetUp() override {
    for (uint32_t i = 0; i < num_vectors; i++) {
      _vectors.push_back(createTestSparseVec());
    }

    std::ofstream output_file =

        dataset::SafeFileIO::ofstream(filename);

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

  void TearDown() override { ASSERT_FALSE(std::remove(filename.c_str())); }

  std::vector<TestSparseVec> _vectors;

  std::mt19937 gen;
  std::uniform_int_distribution<uint32_t> _label_dist;
  std::uniform_int_distribution<uint32_t> _index_dist;
  std::uniform_int_distribution<int32_t> _value_dist;
};

TEST_F(SvmDatasetTestFixture, BoltSvmDatasetTest) {
  auto [data, labels] = SvmDatasetLoader::loadDataset(filename, batch_size);

  // Check data vectors are correct.
  uint32_t vec_count = 0;
  for (const auto& batch : *data) {
    uint32_t batch_size = batch.getBatchSize();
    ASSERT_TRUE(batch_size == batch_size ||
                batch_size == num_vectors % batch_size);

    for (uint32_t v = 0; v < batch_size; v++) {
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
  ASSERT_EQ(vec_count, num_vectors);

  // Check labels are correct.
  uint32_t label_count = 0;
  for (const auto& batch : *labels) {
    ASSERT_TRUE(batch.getBatchSize() == batch_size ||
                batch.getBatchSize() == num_vectors % batch_size);

    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      ASSERT_EQ(batch[v].len, _vectors.at(label_count).labels.size());
      for (uint32_t i = 0; i < batch[v].len; i++) {
        ASSERT_EQ(batch[v].active_neurons[i],
                  _vectors.at(label_count).labels.at(i));
        ASSERT_FLOAT_EQ(batch[v].activations[i],
                        1.0 / _vectors.at(label_count).labels.size());
      }

      label_count++;
    }
  }
  ASSERT_EQ(label_count, num_vectors);
}

}  // namespace thirdai::dataset