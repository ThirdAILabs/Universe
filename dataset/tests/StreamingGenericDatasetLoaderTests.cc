#include "MockBlock.h"
#include <bolt/src/layers/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <tuple>

namespace thirdai::dataset {

class StreamingGenericDatasetLoaderTests : public ::testing::Test {
 public:
  void SetUp() override { writeMockFile(); }

  void TearDown() override { ASSERT_FALSE(remove(MOCK_FILE)); }

  static StreamingGenericDatasetLoader makeMockPipeline(bool shuffle,
                                                        uint32_t seed = 0) {
    auto mock_block =
        std::make_shared<MockBlock>(/* column = */ 0, /* dense = */ true);

    /*
      2 input blocks vs 1 label block to distinguish
      between input and label vectors.
    */
    std::vector<std::shared_ptr<Block>> input_blocks({mock_block, mock_block});
    std::vector<std::shared_ptr<Block>> label_blocks({mock_block});

    return {
        MOCK_FILE,    input_blocks,
        label_blocks, batch_size,
        shuffle,      DatasetShuffleConfig(n_batches_in_shuffle_buffer, seed)};
  }

  static std::tuple<BoltDatasetPtr, BoltDatasetPtr> streamToInMemoryDataset(
      StreamingGenericDatasetLoader&& pipeline) {
    std::vector<bolt::BoltBatch> input_batches;
    std::vector<bolt::BoltBatch> label_batches;
    while (auto batch = pipeline.nextBatchTuple()) {
      auto [input_batch, label_batch] = std::move(batch.value());
      input_batches.push_back(std::move(input_batch));
      label_batches.push_back(std::move(label_batch));
    }
    return {std::make_shared<BoltDataset>(std::move(input_batches)),
            std::make_shared<BoltDataset>(std::move(label_batches))};
  }

  static void assertCorrectVectors(
      std::tuple<BoltDatasetPtr, BoltDatasetPtr>& dataset) {
    const auto& [inputs, labels] = dataset;
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
        found[static_cast<size_t>(input_vec.activations[0])] = true;
      }
    }

    for (auto found_i : found) {
      ASSERT_TRUE(found_i);
    }
  }

  static bool isOrdered(std::tuple<BoltDatasetPtr, BoltDatasetPtr>& dataset) {
    const auto& [inputs, _] = dataset;

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

  static bool sameOrder(std::tuple<BoltDatasetPtr, BoltDatasetPtr>& dataset_1,
                        std::tuple<BoltDatasetPtr, BoltDatasetPtr>& dataset_2) {
    const auto& [inputs_1, _1] = dataset_1;
    const auto& [inputs_2, _2] = dataset_2;

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

  static void assertShuffledEnough(
      std::tuple<BoltDatasetPtr, BoltDatasetPtr>& dataset) {
    const auto& [inputs, _] = dataset;

    // Defined as the number of batches between a vector's
    // original batch and its final batch.
    uint32_t max_vector_displacement = 0;

    for (size_t batch_idx = 0; batch_idx < inputs->numBatches(); batch_idx++) {
      auto& batch = inputs->at(batch_idx);

      /*
        If this shuffling method works then we should
        have a fair share of vectors from this batch,
        from previous batches, and from future batches.
      */

      auto original_vectors_count = countOriginalVectors(batch, batch_idx);
      ASSERT_LE(original_vectors_count,
                0.2 * batch.getBatchSize());  // 0.2 is eyeballed.

      if (batch_idx > 0) {
        ASSERT_TRUE(containsVectorsFromEarlierBatch(batch, batch_idx));
      }

      if (batch_idx < inputs->numBatches() - 1) {
        ASSERT_TRUE(containsVectorsFromLaterBatch(batch, batch_idx));
      }

      max_vector_displacement = std::max(
          max_vector_displacement, getMaxVectorDisplacement(batch, batch_idx));
    }

    /*
      To ensure adequate mixing of samples so that a model
      does not "forget" what earlier samples look like,
      out shuffling mechanism must be able to displace
      vectors further than the length of the shuffle buffer.
    */
    ASSERT_GT(max_vector_displacement, n_batches_in_shuffle_buffer);
    // Sanity check that we have valid measurements.
    size_t n_batches_in_dataset =
        (mock_file_lines + batch_size - 1) / batch_size;
    ASSERT_LT(max_vector_displacement, n_batches_in_dataset);
  }

 private:
  static void writeMockFile() {
    std::ofstream file(MOCK_FILE);
    for (uint32_t i = 0; i < mock_file_lines; i++) {
      file << i << std::endl;
    }
    file.close();
  }

  static size_t countOriginalVectors(bolt::BoltBatch& batch,
                                     uint32_t batch_idx) {
    float original_value_range_start = batch_idx * batch_size;
    float original_value_range_end =
        original_value_range_start + batch.getBatchSize();
    size_t count = 0;

    for (size_t vec_idx = 0; vec_idx < batch.getBatchSize(); vec_idx++) {
      auto value = batch[vec_idx].activations[0];
      if (value >= original_value_range_start &&
          value < original_value_range_end) {
        count++;
      }
    }
    return count;
  }

  static bool containsVectorsFromEarlierBatch(bolt::BoltBatch& batch,
                                              uint32_t batch_idx) {
    float original_value_range_start = batch_idx * batch_size;
    for (size_t vec_idx = 0; vec_idx < batch.getBatchSize(); vec_idx++) {
      auto value = batch[vec_idx].activations[0];
      if (value < original_value_range_start) {
        return true;
      }
    }
    return false;
  }

  static bool containsVectorsFromLaterBatch(bolt::BoltBatch& batch,
                                            uint32_t batch_idx) {
    float original_value_range_end =
        batch_idx * batch_size + batch.getBatchSize();
    for (size_t vec_idx = 0; vec_idx < batch.getBatchSize(); vec_idx++) {
      auto value = batch[vec_idx].activations[0];
      if (value >= original_value_range_end) {
        return true;
      }
    }
    return false;
  }

  static uint32_t getMaxVectorDisplacement(bolt::BoltBatch& batch,
                                           int batch_idx) {
    // Defined as the number of batches between a vector's
    // original batch and its final batch.
    uint32_t max_displacement = 0;

    for (size_t vec_idx = 0; vec_idx < batch.getBatchSize(); vec_idx++) {
      auto value = batch[vec_idx].activations[0];
      int origin_batch_idx = value / batch_size;
      uint32_t displacement = std::abs(origin_batch_idx - batch_idx);
      max_displacement = std::max(displacement, max_displacement);
    }
    return max_displacement;
  }

 protected:
  static constexpr const char* MOCK_FILE = "mock.txt";

  /*
    The last batch will be smaller.
    This tests that our method works even when the last
    batch is smaller.
  */
  static constexpr uint32_t mock_file_lines = 5100;

  static constexpr uint32_t batch_size = 200;

  static constexpr uint32_t n_batches_in_shuffle_buffer = 10;
};

TEST_F(StreamingGenericDatasetLoaderTests, CorrectUnshuffledInMemoryData) {
  auto unshuffled_pipeline = makeMockPipeline(/* shuffle = */ false);
  auto in_memory_data = unshuffled_pipeline.loadInMemory();
  assertCorrectVectors(in_memory_data);
  ASSERT_TRUE(isOrdered(in_memory_data));
}

TEST_F(StreamingGenericDatasetLoaderTests, CorrectUnshuffledStreamedData) {
  auto unshuffled_pipeline = makeMockPipeline(/* shuffle = */ false);
  auto streamed_data = streamToInMemoryDataset(std::move(unshuffled_pipeline));
  assertCorrectVectors(streamed_data);
  ASSERT_TRUE(isOrdered(streamed_data));
}

TEST_F(StreamingGenericDatasetLoaderTests,
       CorrectVectorsInShuffledInMemoryData) {
  auto shuffled_pipeline = makeMockPipeline(/* shuffle = */ true);
  auto in_memory_data = shuffled_pipeline.loadInMemory();
  assertCorrectVectors(in_memory_data);
  ASSERT_FALSE(isOrdered(in_memory_data));
}

TEST_F(StreamingGenericDatasetLoaderTests,
       CorrectVectorsInShuffledStreamedData) {
  auto shuffled_pipeline = makeMockPipeline(/* shuffle = */ true);
  auto streamed_data = streamToInMemoryDataset(std::move(shuffled_pipeline));
  assertCorrectVectors(streamed_data);
  ASSERT_FALSE(isOrdered(streamed_data));
}

TEST_F(StreamingGenericDatasetLoaderTests,
       ShuffledInMemoryDataSameSeedSameOrder) {
  uint32_t seed = 10;
  auto shuffled_pipeline_1 = makeMockPipeline(/* shuffle = */ true, seed);
  auto shuffled_pipeline_2 = makeMockPipeline(/* shuffle = */ true, seed);
  auto in_memory_data_1 = shuffled_pipeline_1.loadInMemory();
  auto in_memory_data_2 = shuffled_pipeline_2.loadInMemory();
  ASSERT_TRUE(sameOrder(in_memory_data_1, in_memory_data_2));
}

TEST_F(StreamingGenericDatasetLoaderTests,
       ShuffledStreamedDataSameSeedSameOrder) {
  uint32_t seed = 10;
  auto shuffled_pipeline_1 = makeMockPipeline(/* shuffle = */ true, seed);
  auto shuffled_pipeline_2 = makeMockPipeline(/* shuffle = */ true, seed);
  auto streamed_data_1 =
      streamToInMemoryDataset(std::move(shuffled_pipeline_1));
  auto streamed_data_2 =
      streamToInMemoryDataset(std::move(shuffled_pipeline_2));
  ASSERT_TRUE(sameOrder(streamed_data_1, streamed_data_2));
}

TEST_F(StreamingGenericDatasetLoaderTests,
       ShuffledInMemoryDataDifferentSeedDifferentOrder) {
  auto shuffled_pipeline_1 =
      makeMockPipeline(/* shuffle = */ true, /* seed = */ 1);
  auto shuffled_pipeline_2 =
      makeMockPipeline(/* shuffle = */ true, /* seed = */ 2);
  auto in_memory_data_1 = shuffled_pipeline_1.loadInMemory();
  auto in_memory_data_2 = shuffled_pipeline_2.loadInMemory();
  ASSERT_FALSE(sameOrder(in_memory_data_1, in_memory_data_2));
}

TEST_F(StreamingGenericDatasetLoaderTests,
       ShuffledStreamedDataDifferentSeedDifferentOrder) {
  auto shuffled_pipeline_1 =
      makeMockPipeline(/* shuffle = */ true, /* seed = */ 1);
  auto shuffled_pipeline_2 =
      makeMockPipeline(/* shuffle = */ true, /* seed = */ 2);
  auto streamed_data_1 =
      streamToInMemoryDataset(std::move(shuffled_pipeline_1));
  auto streamed_data_2 =
      streamToInMemoryDataset(std::move(shuffled_pipeline_2));
  ASSERT_FALSE(sameOrder(streamed_data_1, streamed_data_2));
}

TEST_F(StreamingGenericDatasetLoaderTests,
       ShuffledInMemoryDataIsShuffledEnough) {
  auto unshuffled_pipeline = makeMockPipeline(/* shuffle = */ true);
  auto in_memory_data = unshuffled_pipeline.loadInMemory();
  assertShuffledEnough(in_memory_data);
}

TEST_F(StreamingGenericDatasetLoaderTests,
       ShuffledStreamedDataIsShuffledEnough) {
  auto unshuffled_pipeline = makeMockPipeline(/* shuffle = */ true);
  auto streamed_data = streamToInMemoryDataset(std::move(unshuffled_pipeline));
  assertShuffledEnough(streamed_data);
}

}  // namespace thirdai::dataset
