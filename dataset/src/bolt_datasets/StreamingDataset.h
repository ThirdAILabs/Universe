#pragma once

#include "BatchProcessor.h"
#include "BoltDatasets.h"
#include "DataLoader.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Dataset.h>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <vector>

namespace thirdai::dataset {

template <typename... BATCH_Ts>
class StreamingDataset {
 public:
  StreamingDataset(std::shared_ptr<DataLoader> data_loader,
                   std::shared_ptr<BatchProcessor<BATCH_Ts...>> batch_processor)
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

  std::optional<std::tuple<BATCH_Ts...>> nextBatch() {
    auto rows = _data_loader->nextBatch();
    if (!rows) {
      return std::nullopt;
    }
    auto batch = _batch_processor->createBatch(*rows);

    return batch;
  }

  std::tuple<std::shared_ptr<InMemoryDataset<BATCH_Ts>>...> loadInMemory() {
    std::tuple<std::vector<BATCH_Ts>...> datasets;

    uint64_t len = 0;

    auto start = std::chrono::high_resolution_clock::now();

    while (std::optional<std::tuple<BATCH_Ts...>> batch = nextBatch()) {
      len += std::get<0>(batch.value()).getBatchSize();

      std::apply(
          [&](auto&... lists) {
            std::apply(
                [&](auto&... vals) { (lists.push_back(std::move(vals)), ...); },
                batch.value());
          },
          datasets);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout
        << "Loaded " << len << " vectors from '" + _data_loader->resourceName()
        << "' in "
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
        << " seconds" << std::endl;

    std::tuple<std::shared_ptr<InMemoryDataset<BATCH_Ts>>...> dataset_ptrs =
        std::apply(
            [&](auto&... batch_lists) {
              return std::make_tuple(
                  std::make_shared<InMemoryDataset<BATCH_Ts>>(
                      std::move(batch_lists))...);
            },
            datasets);

    return dataset_ptrs;
  }

  uint32_t getMaxBatchSize() const { return _data_loader->getMaxBatchSize(); }

 private:
  std::shared_ptr<DataLoader> _data_loader;
  std::shared_ptr<BatchProcessor<BATCH_Ts...>> _batch_processor;
};

}  // namespace thirdai::dataset