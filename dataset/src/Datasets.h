#pragma once

#include "BatchProcessor.h"
#include "InMemoryDataset.h"
#include "StreamingDataset.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/batch_processors/ClickThroughBatchProcessor.h>
#include <dataset/src/batch_processors/SvmBatchProcessor.h>
#include <dataset/src/batch_types/BoltTokenBatch.h>
#include <dataset/src/batch_types/ClickThroughBatch.h>
#include <dataset/src/parsers/ClickThroughParser.h>

namespace thirdai::dataset {

using BoltDataset = InMemoryDataset<bolt::BoltBatch>;
using BoltDatasetPtr = std::shared_ptr<BoltDataset>;
using BoltDatasetList = std::vector<BoltDatasetPtr>;

using BoltTokenDataset = InMemoryDataset<BoltTokenBatch>;
using BoltTokenDatasetPtr = std::shared_ptr<BoltTokenDataset>;
using BoltTokenDatasetList = std::vector<BoltTokenDatasetPtr>;

struct SvmDatasetLoader {
  static std::tuple<BoltDatasetPtr, BoltDatasetPtr> loadDataset(
      const std::string& filename, uint32_t batch_size,
      bool softmax_for_multiclass = true) {
    auto batch_processor =
        std::make_shared<SvmBatchProcessor>(softmax_for_multiclass);

    auto dataset =
        StreamingDataset<bolt::BoltBatch, bolt::BoltBatch>::loadDatasetFromFile(
            filename, batch_size, batch_processor);

    return dataset->loadInMemory();
  }
};

struct ClickThroughDatasetLoader {
  static std::tuple<BoltDatasetPtr, BoltTokenDatasetPtr, BoltDatasetPtr>
  loadDataset(const std::string& filename, uint32_t batch_size,
              uint32_t num_dense_features, uint32_t num_categorical_features,
              bool sparse_label_encoding = true) {
    auto batch_processor = std::make_shared<ClickThroughBatchProcessor>(
        num_dense_features, num_categorical_features, sparse_label_encoding);

    auto dataset =
        StreamingDataset<bolt::BoltBatch, BoltTokenBatch,
                         bolt::BoltBatch>::loadDatasetFromFile(filename,
                                                               batch_size,
                                                               batch_processor);

    return dataset->loadInMemory();
  }
};

}  // namespace thirdai::dataset