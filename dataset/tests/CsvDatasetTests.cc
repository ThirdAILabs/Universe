#include <gtest/gtest.h>
#include <dataset/src/Dataset.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace thirdai::dataset {

const std::string filename = "./csv_dataset_test_file";
static const uint32_t num_vectors = 10000, batch_size = 256, seed = 590240;

class CsvDatasetTestFixture : public ::testing::Test {
 public:
  CsvDatasetTestFixture()
      : gen(seed),
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

  void generateTestFile(const char& delimiter) {
    std::ofstream output_file(filename);

    ASSERT_TRUE(output_file.is_open());
    ASSERT_TRUE(output_file.good());
    ASSERT_FALSE(output_file.bad());
    ASSERT_FALSE(output_file.fail());
    for (const auto& vec : _vectors) {
      output_file << vec.label;

      for (const auto& x : vec.values) {
        output_file << delimiter << x;
      }
      output_file << std::endl;
    }

    output_file.close();
  }

  static void generateTestFileWithEntry(const std::string& entry) {
    std::ofstream output_file(filename);

    ASSERT_TRUE(output_file.is_open());
    ASSERT_TRUE(output_file.good());
    ASSERT_FALSE(output_file.bad());
    ASSERT_FALSE(output_file.fail());
    output_file << entry;
    output_file << std::endl;
    output_file.close();
  }

  static void deleteTestFile() { ASSERT_FALSE(std::remove(filename.c_str())); }

  void SetUp() override {
    for (uint32_t i = 0; i < num_vectors; i++) {
      _vectors.push_back(createTestSparseVec());
    }
  }

  void TearDown() override {}

  std::vector<TestSparseVec> _vectors;

  std::mt19937 gen;
  std::uniform_int_distribution<uint32_t> _label_dist;
  std::uniform_int_distribution<int32_t> _value_dist;

 private:
  static const uint32_t _dim = 1000, _num_classes = 10000;
  static const int32_t _val_range = 1000;
};

TEST_F(CsvDatasetTestFixture, InMemoryDatasetTest) {
  for (const char& delimiter : {',', '\t', ' '}) {
    generateTestFile(delimiter);

    InMemoryDataset<DenseBatch> dataset(filename, batch_size,
                                        CsvDenseBatchFactory(delimiter));

    uint32_t vec_count = 0;
    for (const auto& batch : dataset) {
      ASSERT_TRUE(batch.getBatchSize() == batch_size ||
                  batch.getBatchSize() == num_vectors % batch_size);

      for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
        ASSERT_EQ(batch.id(v), vec_count);

        ASSERT_EQ(batch.labels(v).size(), 1);
        for (const auto& label : batch.labels(v)) {
          ASSERT_EQ(label, _vectors.at(vec_count).label);
        }

        ASSERT_EQ(batch[v].dim(), _vectors[vec_count].values.size());
        for (uint32_t i = 0; i < batch[v].dim(); i++) {
          ASSERT_EQ(batch.at(v)._values[i],
                    _vectors.at(vec_count).values.at(i));
        }

        vec_count++;
      }
    }
    ASSERT_EQ(vec_count, num_vectors);

    deleteTestFile();
  }
}

TEST_F(CsvDatasetTestFixture, StreamedDatasetTest) {
  for (const char& delimiter : {',', '\t', ' '}) {
    generateTestFile(delimiter);

    StreamedDataset<DenseBatch> dataset(
        filename, batch_size,
        std::make_unique<CsvDenseBatchFactory>(delimiter));

    uint32_t vec_count = 0;
    while (auto batch_opt = dataset.nextBatch()) {
      const DenseBatch& batch = *batch_opt;
      ASSERT_TRUE(batch.getBatchSize() == batch_size ||
                  batch.getBatchSize() == num_vectors % batch_size);

      for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
        ASSERT_EQ(batch.id(v), vec_count);

        ASSERT_EQ(batch.labels(v).size(), 1);
        for (const auto& label : batch.labels(v)) {
          ASSERT_EQ(label, _vectors.at(vec_count).label);
        }

        ASSERT_EQ(batch[v].dim(), _vectors[vec_count].values.size());
        for (uint32_t i = 0; i < batch[v].dim(); i++) {
          ASSERT_EQ(batch.at(v)._values[i],
                    _vectors.at(vec_count).values.at(i));
        }

        vec_count++;
      }
    }
    ASSERT_EQ(vec_count, num_vectors);

    deleteTestFile();
  }
}

TEST_F(CsvDatasetTestFixture, EmptyValuesTest) {
  std::vector<std::pair<std::string, std::vector<float>>> entries_and_expected{
      {"5,,,", {0, 0, 0}},
      {"5,,5,5", {0, 5, 5}},
  };
  for (const auto& entry_expected_pair : entries_and_expected) {
    generateTestFileWithEntry(entry_expected_pair.first);
    InMemoryDataset<DenseBatch> dataset(filename, batch_size,
                                        CsvDenseBatchFactory(','));
    ASSERT_EQ(dataset.len(), 1);
    for (uint32_t i = 0; i < dataset.at(0).at(0).dim(); ++i) {
      ASSERT_EQ(dataset.at(0).at(0)._values[i], entry_expected_pair.second[i]);
    }
    deleteTestFile();
  }
}

TEST_F(CsvDatasetTestFixture, ErroneousFilesTest) {
  std::vector<std::pair<std::string, std::string>> entries_and_errors{
      {",5",
       "Invalid dataset file: Found a line that doesn't start with a label."},
      {"5", "Invalid dataset file: The line only contains a label."},
      {"5.0.5,5,2", "Invalid dataset file: Found invalid character: ."},
      {"5a5", "Invalid dataset file: Found invalid character: a"},
      {"5,a,5", "Invalid dataset file: Found invalid character: a"},
      {"5,5 ", "Invalid dataset file: Found invalid character:  "},
      {"5 ,5", "Invalid dataset file: Found invalid character:  "},
  };
  for (const auto& entry_error_pair : entries_and_errors) {
    generateTestFileWithEntry(entry_error_pair.first);
    bool failed = false;
    try {
      InMemoryDataset<DenseBatch> dataset(filename, batch_size,
                                          CsvDenseBatchFactory(','));
    } catch (const std::invalid_argument& e) {
      EXPECT_STREQ(entry_error_pair.second.c_str(), e.what());
      failed = true;
    }
    EXPECT_TRUE(failed);
    deleteTestFile();
  }
}

TEST_F(CsvDatasetTestFixture, ErroneousDelimiterTest) {
  for (const char& delimiter :
       {'.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}) {
    bool failed = false;
    try {
      InMemoryDataset<DenseBatch> dataset(filename, batch_size,
                                          CsvDenseBatchFactory(delimiter));
    } catch (const std::invalid_argument& e) {
      EXPECT_STREQ("Invalid delimiter: " + delimiter, e.what());
      failed = true;
    }
    EXPECT_TRUE(failed);
  }
}

}  // namespace thirdai::dataset