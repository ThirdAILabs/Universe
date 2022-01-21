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

const std::string _filename = "./csv_dataset_test_file";
static const uint32_t _num_vectors = 10000, _batch_size = 256;

class CsvDatasetTestFixture : public ::testing::Test {
 public:
  CsvDatasetTestFixture()
      : gen(590240),
        _label_dist(0, _num_classes - 1),
        _value_dist(-_val_range, _val_range) {}

  struct TestSparseVec {
    uint32_t label;
    std::vector<float> values;
  };

  TestSparseVec createTestSparseVec() {
    TestSparseVec vec;

    vec.label = _label_dist(gen);

    for (uint32_t i = 0; i < _dim; i++) {
      vec.values.push_back(_value_dist(gen) * 0.125);
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
      output_file << vec.label;

      uint32_t i = 0;
      for (const auto& x : vec.values) {
        const char* delimiter = ",";
        switch (i++ & 3) {
          case 0:
            break;
          case 1:
            delimiter = ", ";
            break;
          case 2:
            delimiter = "\t,";
            break;
          case 3:
            delimiter = " ,\t ";
        }
        output_file << delimiter << x;
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

  // void TearDown() override { ASSERT_FALSE(std::remove(_filename.c_str())); }

  std::vector<TestSparseVec> _vectors;

  std::mt19937 gen;
  std::uniform_int_distribution<uint32_t> _label_dist;
  std::uniform_int_distribution<int32_t> _value_dist;

 private:
  static const uint32_t _dim = 1000, _num_classes = 10000;
  static const int32_t _val_range = 1000;
};

TEST_F(CsvDatasetTestFixture, InMemoryDatasetTest) {
  InMemoryDataset<DenseBatch> dataset(_filename, _batch_size,
                                      CsvDenseBatchFactory{});

  uint32_t vec_count = 0;
  for (const auto& batch : dataset) {
    ASSERT_TRUE(batch.getBatchSize() == _batch_size ||
                batch.getBatchSize() == _num_vectors % _batch_size);

    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      ASSERT_EQ(batch.id(v), vec_count);

      ASSERT_EQ(batch.labels(v).size(), 1);
      for (auto& label : batch.labels(v)) {
        ASSERT_EQ(label, _vectors.at(vec_count).label);
      }

      ASSERT_EQ(batch[v].dim(), _vectors[vec_count].values.size());
      for (uint32_t i = 0; i < batch[v].dim(); i++) {
        ASSERT_EQ(batch.at(v)._values[i], _vectors.at(vec_count).values.at(i));
      }

      vec_count++;
    }
  }
  ASSERT_EQ(vec_count, _num_vectors);
}

TEST_F(CsvDatasetTestFixture, StreamedDatasetTest) {
  StreamedDataset<DenseBatch> dataset(_filename, _batch_size,
                                      std::make_unique<CsvDenseBatchFactory>());

  uint32_t vec_count = 0;
  while (auto batch_opt = dataset.nextBatch()) {
    const DenseBatch& batch = *batch_opt;
    ASSERT_TRUE(batch.getBatchSize() == _batch_size ||
                batch.getBatchSize() == _num_vectors % _batch_size);

    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      ASSERT_EQ(batch.id(v), vec_count);

      ASSERT_EQ(batch.labels(v).size(), 1);
      for (uint32_t i = 0; i < batch.labels(v).size(); i++) {
        ASSERT_EQ(batch.labels(v).at(i), _vectors.at(vec_count).label);
      }

      ASSERT_EQ(batch[v].dim(), _vectors[vec_count].values.size());
      for (uint32_t i = 0; i < batch[v].dim(); i++) {
        ASSERT_EQ(batch.at(v)._values[i], _vectors.at(vec_count).values.at(i));
      }

      vec_count++;
    }
  }
  ASSERT_EQ(vec_count, _num_vectors);
}

}  // namespace thirdai::dataset