#pragma once

#include "BatchProcessor.h"
#include "BoltDatasets.h"
#include "DataLoader.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Dataset.h>
#include <optional>

namespace thirdai::dataset {

class StreamingDataset {
 public:
  StreamingDataset(std::shared_ptr<DataLoader>& data_loader,
                   std::shared_ptr<BatchProcessor>& batch_processor)
      : _data_loader(data_loader), _batch_processor(batch_processor) {}

  std::optional<BoltDataLabelPair> nextBatch() {
    auto rows = _data_loader->nextBatch();
    if (!rows) {
      return std::nullopt;
    }
    auto batch = _batch_processor->createBatch(*rows);

    return batch;
  }

  DatasetWithLabels loadInMemory() {
    std::vector<bolt::BoltBatch> data;
    std::vector<bolt::BoltBatch> labels;

    uint64_t len = 0;

    while (auto batch = nextBatch()) {
      len += batch->first.getBatchSize();
      data.push_back(std::move(batch->first));
      labels.push_back(std::move(batch->second));
    }

    return DatasetWithLabels(BoltDataset(std::move(data), len),
                             BoltDataset(std::move(labels), len));
  }

  uint32_t getMaxBatchSize() const { return _data_loader->getMaxBatchSize(); }

 private:
  std::shared_ptr<DataLoader> _data_loader;
  std::shared_ptr<BatchProcessor> _batch_processor;
};

}  // namespace thirdai::dataset