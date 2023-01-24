#include "MockBlock.h"
#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/GenericFeaturizer.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>

namespace thirdai::dataset {

class DatasetLoaderTests : public ::testing::Test {
 public:
  // We need this custom setup method to be called at the beginning of every
  // test with a different file name so that we can safely run tests in parallel
  // (if all mock files have the same name, there will be race conditions on
  // the file)
  static void setUp(const std::string& filename) {
    _mock_file_name = filename;
    writeMockFile(filename);
  }

  void TearDown() override { ASSERT_FALSE(remove(_mock_file_name.c_str())); }

  static DatasetLoader makeMockPipeline(bool shuffle, uint32_t seed = 0) {
    auto mock_block =
        std::make_shared<MockBlock>(/* column = */ 0, /* dense = */ true);

    /*
      2 input blocks vs 1 label block to distinguish
      between input and label vectors.
    */
    std::vector<std::shared_ptr<Block>> input_blocks({mock_block, mock_block});
    std::vector<std::shared_ptr<Block>> label_blocks({mock_block});

    auto data_source = std::make_shared<SimpleFileDataSource>(_mock_file_name);

    auto featurizer =
        std::make_shared<GenericFeaturizer>(input_blocks, label_blocks);

    return DatasetLoader(
        data_source, featurizer, shuffle,
        DatasetShuffleConfig(n_batches_in_shuffle_buffer, seed));
  }

  static std::vector<BoltDatasetPtr> streamToInMemoryDataset(
      DatasetLoader&& pipeline) {
    std::vector<BoltBatch> input_batches;
    std::vector<BoltBatch> label_batches;
    while (auto batch = pipeline.streamInMemory(1)) {
      input_batches.push_back(std::move(batch->first.at(0)->at(0)));
      label_batches.push_back(std::move(batch->second->at(0)));
    }
    return {std::make_shared<BoltDataset>(std::move(input_batches)),
            std::make_shared<BoltDataset>(std::move(label_batches))};
  }

  static void assertCorrectVectors(const BoltDatasetPtr& inputs,
                                   const BoltDatasetPtr& labels) {
    std::vector<bool> found(inputs->len());

    for (size_t batch_idx = 0; batch_idx < inputs->numBatches(); batch_idx++) {
      auto& input_batch = inputs->at(batch_idx);
      auto& label_batch = labels->at(batch_idx);
      for (size_t vec_idx = 0; vec_idx < input_batch.getBatchSize();
           vec_idx++) {
        auto& input_vec = input_batch[vec_idx];
        auto& label_vec = label_batch[vec_idx];

        ASSERT_EQ(input_vec.len, 2);
        ASSERT_EQ(label_vec.len, 1);
        ASSERT_EQ(input_vec.activations[0], input_vec.activations[1]);
        ASSERT_EQ(input_vec.activations[0], label_vec.activations[0]);
        found.at(static_cast<size_t>(input_vec.activations[0])) = true;
      }
    }

    for (auto found_i : found) {
      ASSERT_TRUE(found_i);
    }
  }

  static bool isOrdered(const BoltDatasetPtr& inputs) {
    // Expected activation of i-th vector = i.
    float cur_expected_value = 0.0;

    for (size_t batch_idx = 0; batch_idx < inputs->numBatches(); batch_idx++) {
      auto& input_batch = inputs->at(batch_idx);
      for (size_t vec_idx = 0; vec_idx < input_batch.getBatchSize();
           vec_idx++) {
        if (input_batch[vec_idx].activations[0] != cur_expected_value) {
          return false;
        }
        cur_expected_value++;
      }
    }
    return true;
  }

  static bool sameOrder(const BoltDatasetPtr& inputs_1,
                        const BoltDatasetPtr& inputs_2) {
    for (size_t batch_idx = 0; batch_idx < inputs_1->numBatches();
         batch_idx++) {
      auto& input_batch_1 = inputs_1->at(batch_idx);
      auto& input_batch_2 = inputs_2->at(batch_idx);
      for (size_t vec_idx = 0; vec_idx < input_batch_1.getBatchSize();
           vec_idx++) {
        if (input_batch_1[vec_idx].activations[0] !=
            input_batch_2[vec_idx].activations[0]) {
          return false;
        }
      }
    }
    return true;
  }

  static void assertShuffledEnough(const BoltDatasetPtr& inputs) {
    // Defined as the number of batches between a vector's
    // original batch and its final batch.
    uint32_t max_vector_displacement = 0;

    for (size_t batch_idx = 0; batch_idx < inputs->numBatches(); batch_idx++) {
      auto& batch = inputs->at(batch_idx);

      /*
        The probability that a vector stays in its
        original batch is ~roughly~ 1 / number of
        batches in shuffle buffer. We set the length
        of the buffer to 10, so we expect ~10% of the
        batch to be its original contents.
      */
      const float percent_original_vectors_threshold = 0.2;
      auto original_vectors_count = countOriginalVectors(batch, batch_idx);
      ASSERT_LE(original_vectors_count,
                percent_original_vectors_threshold * batch.getBatchSize());

      if (batch_idx > 0) {
        ASSERT_TRUE(containsVectorsFromEarlierBatch(batch, batch_idx));
      }

      if (batch_idx < inputs->numBatches() - 1) {
        ASSERT_TRUE(containsVectorsFromLaterBatch(batch, batch_idx));
      }

      max_vector_displacement = std::max(
          max_vector_displacement, maxVectorDisplacement(batch, batch_idx));
    }

    /*
      To ensure adequate mixing of samples so that a model
      does not "forget" what earlier samples look like,
      our shuffling mechanism must be able to displace
      vectors further than the length of the shuffle buffer.
    */
    ASSERT_GT(max_vector_displacement, n_batches_in_shuffle_buffer);

    // Sanity check
    size_t n_batches_in_dataset =
        (mock_file_lines + batch_size - 1) / batch_size;
    ASSERT_LT(max_vector_displacement, n_batches_in_dataset);
  }

 private:
  static void writeMockFile(const std::string& mock_file_name) {
    std::ofstream file(mock_file_name);
    for (uint32_t i = 0; i < mock_file_lines; i++) {
      file << i << std::endl;
    }
    file.close();
  }

  static size_t countOriginalVectors(BoltBatch& batch, uint32_t batch_idx) {
    float original_value_range_start = batch_idx * batch_size;
    float original_value_range_end =
        original_value_range_start + batch.getBatchSize();
    size_t count = 0;

    for (const auto& vec : batch) {
      auto value = vec.activations[0];
      if (value >= original_value_range_start &&
          value < original_value_range_end) {
        count++;
      }
    }
    return count;
  }

  static bool containsVectorsFromEarlierBatch(BoltBatch& batch,
                                              uint32_t batch_idx) {
    float original_value_range_start = batch_idx * batch_size;
    return std::any_of(batch.begin(), batch.end(), [&](const auto& vec) {
      return vec.activations[0] < original_value_range_start;
    });
  }

  static bool containsVectorsFromLaterBatch(BoltBatch& batch,
                                            uint32_t batch_idx) {
    float original_value_range_end =
        batch_idx * batch_size + batch.getBatchSize();
    return std::any_of(batch.begin(), batch.end(), [&](const auto& vec) {
      return vec.activations[0] >= original_value_range_end;
    });
  }

  static uint32_t maxVectorDisplacement(BoltBatch& batch, int batch_idx) {
    // Defined as the number of batches between a vector's
    // original batch and its final batch.
    uint32_t max_displacement = 0;

    for (const auto& vec : batch) {
      auto value = vec.activations[0];
      int origin_batch_idx = value / batch_size;
      uint32_t displacement = std::abs(origin_batch_idx - batch_idx);
      max_displacement = std::max(displacement, max_displacement);
    }
    return max_displacement;
  }

 protected:
  inline static std::string _mock_file_name;

  /*
    The last batch will be smaller.
    This tests that our method works even when the last
    batch is smaller.
  */
  static constexpr uint32_t mock_file_lines = 10100;

  static constexpr uint32_t batch_size = 200;

  static constexpr uint32_t n_batches_in_shuffle_buffer = 10;
};

TEST_F(DatasetLoaderTests, CorrectUnshuffledInMemoryData) {
  DatasetLoaderTests::setUp("mock0.txt");
  auto unshuffled_pipeline = makeMockPipeline(/* shuffle = */ false);
  auto in_memory_data = unshuffled_pipeline.loadInMemory();
  assertCorrectVectors(in_memory_data.first.at(0), in_memory_data.second);
  ASSERT_TRUE(isOrdered(in_memory_data.first.at(0)));
}

TEST_F(DatasetLoaderTests, CorrectUnshuffledStreamedData) {
  DatasetLoaderTests::setUp("mock1.txt");
  auto unshuffled_pipeline = makeMockPipeline(/* shuffle = */ false);
  auto streamed_data = streamToInMemoryDataset(std::move(unshuffled_pipeline));
  assertCorrectVectors(streamed_data.at(0), streamed_data.at(1));
  ASSERT_TRUE(isOrdered(streamed_data.at(0)));
}

TEST_F(DatasetLoaderTests, CorrectVectorsInShuffledInMemoryData) {
  DatasetLoaderTests::setUp("mock2.txt");
  auto shuffled_pipeline = makeMockPipeline(/* shuffle = */ true);
  auto in_memory_data = shuffled_pipeline.loadInMemory();
  assertCorrectVectors(in_memory_data.first.at(0), in_memory_data.second);
  ASSERT_FALSE(isOrdered(in_memory_data.first.at(0)));
}

TEST_F(DatasetLoaderTests, CorrectVectorsInShuffledStreamedData) {
  DatasetLoaderTests::setUp("mock3.txt");
  auto shuffled_pipeline = makeMockPipeline(/* shuffle = */ true);
  auto streamed_data = streamToInMemoryDataset(std::move(shuffled_pipeline));
  assertCorrectVectors(streamed_data.at(0), streamed_data.at(1));
  ASSERT_FALSE(isOrdered(streamed_data.at(0)));
}

TEST_F(DatasetLoaderTests, ShuffledInMemoryDataSameSeedSameOrder) {
  DatasetLoaderTests::setUp("mock4.txt");
  uint32_t seed = 10;
  auto shuffled_pipeline_1 = makeMockPipeline(/* shuffle = */ true, seed);
  auto shuffled_pipeline_2 = makeMockPipeline(/* shuffle = */ true, seed);
  auto in_memory_data_1 = shuffled_pipeline_1.loadInMemory();
  auto in_memory_data_2 = shuffled_pipeline_2.loadInMemory();
  ASSERT_TRUE(
      sameOrder(in_memory_data_1.first.at(0), in_memory_data_2.first.at(0)));
}

TEST_F(DatasetLoaderTests, ShuffledStreamedDataSameSeedSameOrder) {
  DatasetLoaderTests::setUp("mock5.txt");
  uint32_t seed = 10;
  auto shuffled_pipeline_1 = makeMockPipeline(/* shuffle = */ true, seed);
  auto shuffled_pipeline_2 = makeMockPipeline(/* shuffle = */ true, seed);
  auto streamed_data_1 =
      streamToInMemoryDataset(std::move(shuffled_pipeline_1));
  auto streamed_data_2 =
      streamToInMemoryDataset(std::move(shuffled_pipeline_2));
  ASSERT_TRUE(sameOrder(streamed_data_1.at(0), streamed_data_2.at(0)));
}

TEST_F(DatasetLoaderTests, ShuffledInMemoryDataDifferentSeedDifferentOrder) {
  DatasetLoaderTests::setUp("mock6.txt");
  auto shuffled_pipeline_1 =
      makeMockPipeline(/* shuffle = */ true, /* seed = */ 1);
  auto shuffled_pipeline_2 =
      makeMockPipeline(/* shuffle = */ true, /* seed = */ 2);
  auto in_memory_data_1 = shuffled_pipeline_1.loadInMemory();
  auto in_memory_data_2 = shuffled_pipeline_2.loadInMemory();
  ASSERT_FALSE(
      sameOrder(in_memory_data_1.first.at(0), in_memory_data_2.first.at(0)));
}

TEST_F(DatasetLoaderTests, ShuffledStreamedDataDifferentSeedDifferentOrder) {
  DatasetLoaderTests::setUp("mock7.txt");
  auto shuffled_pipeline_1 =
      makeMockPipeline(/* shuffle = */ true, /* seed = */ 1);
  auto shuffled_pipeline_2 =
      makeMockPipeline(/* shuffle = */ true, /* seed = */ 2);
  auto streamed_data_1 =
      streamToInMemoryDataset(std::move(shuffled_pipeline_1));
  auto streamed_data_2 =
      streamToInMemoryDataset(std::move(shuffled_pipeline_2));
  ASSERT_FALSE(sameOrder(streamed_data_1.at(0), streamed_data_2.at(0)));
}

TEST_F(DatasetLoaderTests, ShuffledInMemoryDataIsShuffledEnough) {
  DatasetLoaderTests::setUp("mock8.txt");
  auto unshuffled_pipeline = makeMockPipeline(/* shuffle = */ true);
  auto in_memory_data = unshuffled_pipeline.loadInMemory();
  assertShuffledEnough(in_memory_data.first.at(0));
}

TEST_F(DatasetLoaderTests, ShuffledStreamedDataIsShuffledEnough) {
  DatasetLoaderTests::setUp("mock9.txt");
  auto unshuffled_pipeline = makeMockPipeline(/* shuffle = */ true);
  auto streamed_data = streamToInMemoryDataset(std::move(unshuffled_pipeline));
  assertShuffledEnough(streamed_data.at(0));
}

}  // namespace thirdai::dataset
