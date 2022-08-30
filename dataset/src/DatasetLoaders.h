#pragma once

#include "BatchProcessor.h"
#include "Datasets.h"
#include "InMemoryDataset.h"
#include "StreamingDataset.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/batch_processors/ClickThroughBatchProcessor.h>
#include <dataset/src/batch_processors/SvmBatchProcessor.h>

namespace thirdai::dataset {

struct SvmDatasetLoader {
  static std::tuple<BoltDatasetPtr, BoltDatasetPtr> loadDataset(
      const std::string& filename, uint32_t batch_size,
      bool softmax_for_multiclass = true) {
    auto batch_processor =
        std::make_shared<SvmBatchProcessor>(softmax_for_multiclass);

    auto dataset = StreamingDataset<BoltBatch, BoltBatch>::loadDatasetFromFile(
        filename, batch_size, batch_processor);

    return dataset->loadInMemory();
  }
};

struct ClickThroughDatasetLoader {
  static std::tuple<BoltDatasetPtr, BoltDatasetPtr, BoltDatasetPtr> loadDataset(
      const std::string& filename, uint32_t batch_size,
      uint32_t num_dense_features, uint32_t max_num_categorical_features,
      char delimiter) {
    auto batch_processor = std::make_shared<ClickThroughBatchProcessor>(
        num_dense_features, max_num_categorical_features, delimiter);

    auto dataset =
        StreamingDataset<BoltBatch, BoltBatch, BoltBatch>::loadDatasetFromFile(
            filename, batch_size, batch_processor);

    return dataset->loadInMemory();
  }
};

}  // namespace thirdai::dataset