#include "networks/DLRM.h"
#include <dataset/src/Dataset.h>
#include <chrono>
#include <iostream>
#include <vector>

using ClickThroughDataset =
    thirdai::dataset::InMemoryDataset<thirdai::dataset::ClickThroughBatch>;

ClickThroughDataset loadClickThorughDataset(const std::string& filename,
                                            uint32_t batch_size,
                                            uint32_t num_dense_features,
                                            uint32_t num_categorical_features) {
  auto start = std::chrono::high_resolution_clock::now();
  thirdai::dataset::ClickThroughBatchFactory factory(num_dense_features,
                                                     num_categorical_features);
  ClickThroughDataset data(filename, batch_size, std::move(factory));
  auto end = std::chrono::high_resolution_clock::now();
  std::cout
      << "Read " << data.len() << " vectors in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;
  return data;
}

int main() {
  auto train = loadClickThorughDataset(
      "/Users/nmeisburger/ThirdAI/data/intent/train_shuf_criteo.txt", 4, 512,
      6);
  auto test = loadClickThorughDataset(
      "/Users/nmeisburger/ThirdAI/data/intent/test_shuf_criteo.txt", 256, 512,
      26);

  auto embedding = thirdai::bolt::EmbeddingLayerConfig(8, 16, 15);

  std::vector<thirdai::bolt::FullyConnectedLayerConfig> bottom_mlp = {
      {1000, 0.2, "ReLU", {3, 128, 9, 10}}, {100, "ReLU"}};

  std::vector<thirdai::bolt::FullyConnectedLayerConfig> top_mlp = {
      {100, "ReLU"}, {1000, 0.2, "ReLU", {3, 128, 9, 10}}, {151, "Softmax"}};

  thirdai::bolt::DLRM dlrm(embedding, bottom_mlp, top_mlp, 512);

  for (uint32_t e = 0; e < 10; e++) {
    dlrm.train(train, 0.001, 1, 5000, 50000);

    float* scores = new float[151 * test.len()];
    dlrm.predict(test, scores);
  }
  return 0;
}