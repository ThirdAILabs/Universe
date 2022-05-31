#pragma once

#include "BatchProcessor.h"
#include "BoltDatasets.h"
#include "DataLoader.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Dataset.h>
#include <optional>
#include <stdexcept>

namespace thirdai::dataset {

template <typename BATCH_T>
class StreamingDataset {
 public:
  StreamingDataset(std::shared_ptr<DataLoader> data_loader,
                   std::shared_ptr<BatchProcessor<BATCH_T>> batch_processor)
      : _data_loader(std::move(data_loader)),
        _batch_processor(std::move(batch_processor)) {
    if (_batch_processor->expectsHeader()) {
      auto header = _data_loader->getHeader();
      if (!header) {
        throw std::invalid_argument("Cannot read empty file.");
      }
      _batch_processor->processHeader(*header);
    }
  }

  std::optional<BoltDataLabelPair<BATCH_T>> nextBatch() {
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
  std::shared_ptr<BatchProcessor<BATCH_T>> _batch_processor;
};

}  // namespace thirdai::dataset