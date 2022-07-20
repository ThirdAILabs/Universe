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

  std::optional<std::tuple<BATCH_Ts...>> nextBatchTuple() {
    auto rows = _data_loader->nextBatch();
    if (!rows) {
      return std::nullopt;
    }

    return _batch_processor->createBatch(*rows);
  }

  // This function maps the tuple of batches returned by nextBatch() into a
  // tuple of datasets where each dataset contains a list of batches of the type
  // corresponding to that element of the tuple.
  std::tuple<std::shared_ptr<InMemoryDataset<BATCH_Ts>>...> loadInMemory() {
    std::tuple<std::vector<BATCH_Ts>...> batch_lists;

    uint64_t len = 0;

    auto start = std::chrono::high_resolution_clock::now();

    while (auto batch_tuple = nextBatchTuple()) {
      len += std::get<0>(batch_tuple.value()).getBatchSize();

      /**
       * std::apply allows for a tuple to be applied to function that accepts a
       * variadic template argument. We use this here to pass the tuple of
       * vectors that we accumulate the batches in along with the tuple
       * containing the next batches into a variadic template function that
       * calls vector.push_back(...).
       *
       * Helpful stack overflow post about doing this:
       * https://stackoverflow.com/questions/53305395/how-to-create-a-tuple-of-vectors-of-type-deduced-from-a-variadic-template-in-c
       */
      std::apply(
          [&batch_tuple](auto&... batch_lists_arg) {
            std::apply(
                [&](auto&... batch_tuple_arg) {
                  (batch_lists_arg.push_back(std::move(batch_tuple_arg)), ...);
                },
                batch_tuple.value());
          },
          batch_lists);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout
        << "Loaded " << len << " vectors from '" + _data_loader->resourceName()
        << "' in "
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
        << " seconds" << std::endl;

    // We use std::apply again here to call a function acception a variadic
    // template that maps each vector of batches to an InMemoryDataset.
    std::tuple<std::shared_ptr<InMemoryDataset<BATCH_Ts>>...> dataset_ptrs =
        std::apply(
            [](auto&... batch_lists_arg) {
              return std::make_tuple(
                  std::make_shared<InMemoryDataset<BATCH_Ts>>(
                      std::move(batch_lists_arg))...);
            },
            batch_lists);

    return dataset_ptrs;
  }

  uint32_t getMaxBatchSize() const { return _data_loader->getMaxBatchSize(); }

  static std::shared_ptr<StreamingDataset<BATCH_Ts...>> loadDatasetFromFile(
      const std::string& filename, uint32_t batch_size,
      std::shared_ptr<BatchProcessor<BATCH_Ts...>> batch_processor) {
    auto data_loader =
        std::make_shared<SimpleFileDataLoader>(filename, batch_size);

    auto dataset = std::make_shared<StreamingDataset<BATCH_Ts...>>(
        std::move(data_loader), std::move(batch_processor));

    return dataset;
  }

 private:
  std::shared_ptr<DataLoader> _data_loader;
  std::shared_ptr<BatchProcessor<BATCH_Ts...>> _batch_processor;
};

}  // namespace thirdai::dataset