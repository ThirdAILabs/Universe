#pragma once

#include "BatchProcessor.h"
#include "DataSource.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/InMemoryDataset.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <utils/Logging.h>
#include <chrono>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace thirdai::dataset {

class StreamingDataset {
 public:
  StreamingDataset(std::shared_ptr<DataSource> data_source,
                   std::shared_ptr<GenericBatchProcessor> batch_processor)
      : _data_source(std::move(data_source)),
        _batch_processor(std::move(batch_processor)) {
    // Different formats of data may or may not contain headers. Thus we
    // delegate to the particular batch processor to determine if a header is
    // needed. The first row is interpreted as the header. The batch processor
    // is responsible for checking that the header is properly formatted.
    if (_batch_processor->expectsHeader()) {
      auto header = _data_source->nextLine();
      if (!header) {
        throw std::invalid_argument("Cannot read empty file.");
      }
      _batch_processor->processHeader(*header);
    }
  }

  virtual std::optional<std::vector<BoltBatch>> nextBatchVector() {
    auto rows = _data_source->nextBatch();
    if (!rows) {
      return std::nullopt;
    }

    // TODO(Someone): Change these next few lines when batch processor returns 
    // an optional of a vector of BoltBatches
    auto optional_tuple = _batch_processor->createBatch(*rows);
    std::vector<BoltBatch> result;
    result.push_back(std::move(std::get<0>(optional_tuple)));
    result.push_back(std::move(std::get<1>(optional_tuple)));
    return result;
  }

  virtual std::vector<std::shared_ptr<InMemoryDataset<BoltBatch>>>
  loadInMemory() {
    auto datasets = loadInMemory(std::numeric_limits<uint64_t>::max());
    if (!datasets) {
      throw std::invalid_argument("Cannot load datasets from empty resource '" +
                                  _data_source->resourceName() + "'.");
    }
    return datasets.value();
  }

  // This function maps the tuple of batches returned by nextBatch() into a
  // tuple of datasets where each dataset contains a list of batches of the
  // type corresponding to that element of the tuple. NOLINTNEXTLINE
  std::optional<std::vector<BoltDatasetPtr>>
  loadInMemory(uint64_t max_batches) {
    std::vector<std::vector<BoltBatch>> batch_lists;

    uint64_t len = 0;
    uint64_t loaded_batches = 0;

    auto start = std::chrono::high_resolution_clock::now();

    while (auto batch_vector = nextBatchVector()) {
      
      len += batch_vector->at(0).getBatchSize();

      batch_lists.push_back(std::move(*batch_vector));
    
      loaded_batches++;
      if (loaded_batches >= max_batches) {
        break;
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    logging::info(
        "Loaded {} vectors from '{}' in {} seconds.", len,
        _data_source->resourceName(),
        std::chrono::duration_cast<std::chrono::seconds>(end - start).count());

    if (batch_lists.empty()) {
      return std::nullopt;
    }

    std::vector<BoltDatasetPtr> dataset_ptrs;
    for (uint32_t dataset_id = 0; dataset_id < batch_lists.at(0).size(); dataset_id++) {
      std::vector<BoltBatch> dataset_batches;
      dataset_batches.reserve(batch_lists.size());
for (auto & batch_list : batch_lists) {
        dataset_batches.push_back(std::move(batch_list.at(dataset_id)));
      }
      dataset_ptrs.emplace_back(dataset_batches);
    }

    return dataset_ptrs;
  }

  uint32_t getMaxBatchSize() const { return _data_source->getMaxBatchSize(); }

  virtual void restart() {
    _data_source->restart();

    // When we restart we need to make sure we don't reread the header. s
    if (_batch_processor->expectsHeader()) {
      auto header = _data_source->nextLine();
      if (!header) {
        throw std::invalid_argument("Cannot read empty file.");
      }
    }
  }

  static std::shared_ptr<StreamingDataset> loadDataset(
      std::shared_ptr<DataSource> data_source,
      std::shared_ptr<GenericBatchProcessor> batch_processor) {
    auto dataset = std::make_shared<StreamingDataset>(
        std::move(data_source), std::move(batch_processor));

    return dataset;
  }

  virtual ~StreamingDataset() = default;

 protected:
  std::shared_ptr<DataSource> _data_source;

 private:
  std::shared_ptr<GenericBatchProcessor> _batch_processor;
};

}  // namespace thirdai::dataset
