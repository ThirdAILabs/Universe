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
      "/Users/nmeisburger/files/Research/data/mini_criteo.txt", 256, 13, 26);

  auto embedding = thirdai::bolt::EmbeddingLayerConfig(8, 16, 15);

  std::vector<thirdai::bolt::FullyConnectedLayerConfig> bottom_mlp = {
      {1000, 0.2, "ReLU", {3, 128, 9, 10}}, {100, "ReLU"}};

  std::vector<thirdai::bolt::FullyConnectedLayerConfig> top_mlp = {
      {100, "ReLU"}, {1000, 0.2, "ReLU", {3, 128, 9, 10}}, {1, "MeanSquared"}};

  thirdai::bolt::DLRM dlrm(embedding, bottom_mlp, top_mlp, 13);

  dlrm.train(train, 0.0001, 2, 300, 500);
  dlrm.train(train, 0.0001, 2, 300, 500);

  return 0;
}