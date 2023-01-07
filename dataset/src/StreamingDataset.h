#pragma once

#include "BatchProcessor.h"
#include "DataSource.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/InMemoryDataset.h>
#include <utils/Logging.h>
#include <chrono>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace thirdai::dataset {

// This structure uses template magic to deduce std::tuple<BoltBatch...> from
// std::tuple<BATCH_Ts>, where `BATCH_Ts` is unknown until templated.
// Essentially, this deduces how many BATCH_Ts are there and creates a tuple
// with the same number of BoltDatasetPtrs, outputing the result at type.
//
// Once deduced, we can use this to simplify API at loadInMemory(), removing
// template cruft.
//
// This is a stop-gap solution to drop templating over BATCH_T in
// InMemoryDataset. Should be removed in the future as more cleanup happens
// later.
template <typename T, size_t N>
class generate_tuple_type {
  template <typename = std::make_index_sequence<N>>
  struct impl;

  template <size_t... Is>
  struct impl<std::index_sequence<Is...>> {
    template <size_t>
    using wrap = T;

    using type = std::tuple<wrap<Is>...>;
  };

 public:
  using type = typename impl<>::type;
};

template <typename... BATCH_Ts>
class StreamingDataset {
 public:
  // Deduce DatasetTuple type from BATCH_Ts.
  using DatasetTuple =
      typename generate_tuple_type<BoltDatasetPtr, sizeof...(BATCH_Ts)>::type;

  StreamingDataset(std::shared_ptr<DataSource> data_source,
                   std::shared_ptr<BatchProcessor<BATCH_Ts...>> batch_processor)
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

  virtual std::optional<std::tuple<BATCH_Ts...>> nextBatchTuple() {
    auto rows = _data_source->nextBatch();
    if (!rows) {
      return std::nullopt;
    }

    return _batch_processor->createBatch(*rows);
  }

  virtual DatasetTuple loadInMemory() {
    auto datasets = loadInMemory(std::numeric_limits<uint64_t>::max());
    return datasets;
  }

  // This function maps the tuple of batches returned by nextBatch() into a
  // tuple of datasets where each dataset contains a list of batches of the
  // type corresponding to that element of the tuple. NOLINTNEXTLINE
  DatasetTuple loadInMemory(uint64_t max_batches) {
    std::tuple<std::vector<BATCH_Ts>...> batch_lists;

    uint64_t len = 0;
    uint64_t loaded_batches = 0;

    auto start = std::chrono::high_resolution_clock::now();

    while (auto batch_tuple = nextBatchTuple()) {
      len += std::get<0>(batch_tuple.value()).getBatchSize();

      /**
       * std::apply allows for a tuple to be applied to function that accepts
       * a variadic template argument. We use this here to pass the tuple of
       * vectors that we accumulate the batches in along with the tuple
       * containing the next batches into a variadic template function that
       * calls vector.push_back(...).
       *
       * Helpful stack overflow post about doing this:
       * https://stackoverflow.com/questions/53305395/how-to-create-a-tuple-of-vectors-of-type-deduced-from-a-variadic-template-in-c
       */
      auto callback = [&batch_tuple](auto&... batch_lists_arg) {
        std::apply(
            [&](auto&... batch_tuple_arg) {
              (..., batch_lists_arg.push_back(std::move(batch_tuple_arg)));
            },
            batch_tuple.value());
      };

      std::apply(callback, batch_lists);

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

    if (std::get<0>(batch_lists).empty()) {
      throw std::invalid_argument("Cannot load datasets from empty resource '" +
                                  _data_source->resourceName() + "'.");
    }

    // We use std::apply again here to call a function acception a variadic
    // template that maps each vector of batches to an InMemoryDataset.
    auto callback = [](auto&... batch_lists_arg) {
      return std::make_tuple(
          std::make_shared<InMemoryDataset>(std::move(batch_lists_arg))...);
    };

    DatasetTuple dataset_ptrs = std::apply(callback, batch_lists);
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

  static std::shared_ptr<StreamingDataset<BATCH_Ts...>> loadDataset(
      std::shared_ptr<DataSource> data_source,
      std::shared_ptr<BatchProcessor<BATCH_Ts...>> batch_processor) {
    auto dataset = std::make_shared<StreamingDataset<BATCH_Ts...>>(
        std::move(data_source), std::move(batch_processor));

    return dataset;
  }

  virtual ~StreamingDataset() = default;

 protected:
  std::shared_ptr<DataSource> _data_source;

 private:
  std::shared_ptr<BatchProcessor<BATCH_Ts...>> _batch_processor;
};

}  // namespace thirdai::dataset
