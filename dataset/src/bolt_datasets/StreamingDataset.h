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
    // Different formats of data may or may not contain headers. Thus we
    // delegate to the particular batch processor to determine if a header is
    // needed. The first row is interpreted as the header. The batch processor
    // is responsible for checking that the header is properly formatted.
    if (_batch_processor->expectsHeader()) {
      auto header = _data_loader->getHeader();
      if (!header) {
        throw std::invalid_argument("Cannot read empty file.");
      }
      _batch_processor->processHeader(*header);
    }
  }

  virtual std::optional<BoltDataLabelPair<BATCH_T>> nextBatch() {
    auto rows = _data_loader->nextBatch();
    if (!rows) {
      return std::nullopt;
    }
    auto batch = _batch_processor->createBatch(*rows);

    return batch;
  }

  virtual std::pair<std::shared_ptr<InMemoryDataset<BATCH_T>>, BoltDatasetPtr>
  loadInMemory() {
    std::vector<BATCH_T> data;
    std::vector<bolt::BoltBatch> labels;

    uint64_t len = 0;

    while (auto batch = nextBatch()) {
      len += batch->first.getBatchSize();
      data.push_back(std::move(batch->first));
      labels.push_back(std::move(batch->second));
    }

    return {std::make_shared<InMemoryDataset<BATCH_T>>(std::move(data), len),
            std::make_shared<BoltDataset>(std::move(labels), len)};
  }

  virtual uint32_t getMaxBatchSize() const { return _data_loader->getMaxBatchSize(); }

 private:
  std::shared_ptr<DataLoader> _data_loader;
  std::shared_ptr<BatchProcessor<BATCH_T>> _batch_processor;
};

}  // namespace thirdai::dataset