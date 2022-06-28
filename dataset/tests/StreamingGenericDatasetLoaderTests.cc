#include "MockBlock.h"
#include <gtest/gtest.h>
#include <dataset/src/bolt_datasets/StreamingGenericDatasetLoader.h>
#include <cstdio>
#include <fstream>
#include <memory>

namespace thirdai::dataset {

static constexpr const char* MOCK_FILE = "mock.txt";

static void writeMockFile() {
  std::ofstream file(MOCK_FILE);
  for (uint32_t i = 0; i < 1000; i++) {
    file << i << std::endl;
  }
  file.close();
}

static void deleteMockFile() { remove(MOCK_FILE); }

static StreamingGenericDatasetLoader makeMockPipeline(bool shuffle) {
  auto mock_block =
      std::make_shared<MockBlock>(/* column = */ 0, /* dense = */ true);
  return {/* filename = */ MOCK_FILE,
          /* input_blocks = */ {mock_block},
          /* label_blocks = */ {mock_block},
          /* batch_size = */ 10,
          /* shuffle = */ shuffle};
}

TEST(StreamingGenericDatasetLoaderTests, CanShuffle) {
  writeMockFile();
  auto unshuffled_pipeline = makeMockPipeline(/* shuffle = */ false);
  size_t line = 0;
  while (auto batch = unshuffled_pipeline.nextBatch()) {
    for (size_t i = 0; i < batch->first.getBatchSize(); i++) {
      ASSERT_EQ(batch->first[i].activations[0], static_cast<float>(line));
      ASSERT_EQ(batch->second[i].activations[0], static_cast<float>(line));
      line++;
    }
  }
  auto shuffled_pipeline = makeMockPipeline(/* shuffle = */ true);
  line = 0;
  size_t n_under_100 = 0;
  while (auto batch = shuffled_pipeline.nextBatch()) {
    for (size_t i = 0; i < batch->first.getBatchSize(); i++) {
      ASSERT_EQ(batch->first[i].activations[0],
                batch->second[i].activations[0]);
      if (batch->first[i].activations[0] < 100 && line < 30) {
        n_under_100++;
      }
      line++;
    }
  }
  ASSERT_LE(n_under_100, 20);
  ASSERT_EQ(line, 1000);
  deleteMockFile();
}

}  // namespace thirdai::dataset
