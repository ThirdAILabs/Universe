#include "DatasetLoader.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/Datasets.h>
#include <utils/Logging.h>
#include <limits>
#include <utility>

namespace thirdai::dataset {

DatasetLoader::DatasetLoader(DataSourcePtr data_source,
                             dataset::FeaturizerPtr featurizer, bool shuffle,
                             DatasetShuffleConfig shuffle_config,
                             size_t internal_featurization_batch_size)
    : _data_source(std::move(data_source)),
      _featurizer(std::move(featurizer)),
      _buffer_size(shuffle_config.min_buffer_size),
      _batcher(_featurizer->getNumDatasets(), shuffle, shuffle_config.seed),
      _featurization_batch_size(internal_featurization_batch_size) {
  // Different formats of data may or may not contain headers. Thus we
  // delegate to the particular featurizer to determine if a header is
  // needed. The first row is interpreted as the header. The featurizer
  // is responsible for checking that the header is properly formatted.
  if (_featurizer->expectsHeader()) {
    auto header = _data_source->nextLine();
    if (!header) {
      throw std::invalid_argument("Cannot read empty file.");
    }
    _featurizer->processHeader(*header);
  }
}

std::vector<BoltDatasetPtr> DatasetLoader::loadAll(size_t batch_size,
                                                   bool verbose) {
  auto datasets =
      loadSome(/* batch_size = */ batch_size,
               /* num_batches = */ std::numeric_limits<size_t>::max(),
               /* verbose = */ verbose);
  if (!datasets) {
    throw std::invalid_argument(
        "Did not find any data to load from the data source.");
  }
  return datasets.value();
}

std::optional<std::vector<BoltDatasetPtr>> DatasetLoader::loadSome(
    size_t batch_size, size_t num_batches, bool verbose) {
#if THIRDAI_EXPOSE_ALL
  if (verbose) {
    // This is useful internally but we don't want to expose it to keep the
    // output clear and simple.
    std::cout << "loading data | source '" << _data_source->resourceName()
              << "'" << std::endl;
  }
#endif

  auto start = std::chrono::high_resolution_clock::now();

  // We fill the buffer with num_batches * batch_size + _buffer_size vectors
  // so that after exporting num_batches from the buffer we still have
  // _buffer_size vectors left for future shuffling.
  // We also must check if anything in this multiplication and sum overflows and
  // use std::numeric_limits<size_t>::max() in that case, since sometimes (when
  // we want to load everything) we pass in std::numeric_limits<size_t>::max()
  // as num_batches, which would make the expression overflow
  bool will_overflow =
      (std::numeric_limits<size_t>::max() / num_batches <= batch_size) ||
      (std::numeric_limits<size_t>::max() - _buffer_size <=
       num_batches * batch_size);
  size_t fill_size = will_overflow ? std::numeric_limits<size_t>::max()
                                   : num_batches * batch_size + _buffer_size;
  fillVectorBuffer(fill_size);

  auto dataset_slices = _batcher.pop(/* max_num_batches = */ num_batches,
                                     /* target_batch_size = */ batch_size);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

  if (dataset_slices.at(0).empty()) {
#if THIRDAI_EXPOSE_ALL
    if (verbose) {
      // This is to ensure that it always prints complete if it prints that it
      // has started loading above.
      std::cout << "loading data | source '" << _data_source->resourceName()
                << "' | vectors 0 | batches 0 | time " << duration
                << "s | complete\n"
                << std::endl;
    }
#endif
    return std::nullopt;
  }

  std::vector<BoltDatasetPtr> data;
  data.reserve(dataset_slices.size());
  for (auto& dataset_slice : dataset_slices) {
    data.push_back(std::make_shared<BoltDataset>(std::move(dataset_slice)));
  }

  if (verbose) {
    std::cout << "loaded data | source '" << _data_source->resourceName()
              << "' | vectors " << data.at(0)->len() << " | batches "
              << data.at(0)->numBatches() << " | time " << duration
              << "s | complete\n"
              << std::endl;
  }
  return data;
}

void DatasetLoader::restart() {
  _data_source->restart();

  // When we restart we need to make sure we don't reread the header.
  if (_featurizer->expectsHeader()) {
    auto header = _data_source->nextLine();
    if (!header) {
      throw std::invalid_argument("Cannot read empty file.");
    }
  }
}

void DatasetLoader::fillVectorBuffer(size_t num_rows) {
  while (_batcher.size() <= num_rows) {
    auto rows = _data_source->nextBatch(
        /* target_batch_size = */ _featurization_batch_size);
    if (!rows) {
      return;
    }

    auto batch = _featurizer->featurize(*rows);
    _batcher.add(std::move(batch));
  }
}

// std::vector<DatasetSlice> DatasetLoader::popFromBuffer(
//     size_t target_num_batches, size_t target_batch_size) {
//   size_t num_datasets = _featurizer->getNumDatasets();
//   std::vector<std::vector<BoltBatch>> batches(num_datasets);

//   for (size_t batch_id = 0; batch_id < target_num_batches; batch_id++) {
//     // The ith element in this list contains BoltVectors corresponding to the
//     // ith Dataset this DatasetLoader is loading
//     std::vector<std::vector<BoltVector>> batch(num_datasets);
//     for (size_t vec_id = 0; vec_id < target_batch_size; vec_id++) {
//       if (auto next_vectors = _buffer.pop()) {
//         assert(next_vectors->size() == num_datasets);
//         for (size_t dataset_id = 0; dataset_id < num_datasets; dataset_id++)
//         {
//           batch.at(dataset_id).push_back(next_vectors->at(dataset_id));
//         }
//       }
//     }

//     if (batch.at(0).empty()) {
//       break;
//     }

//     for (size_t dataset_id = 0; dataset_id < batch.size(); dataset_id++) {
//       batches.at(dataset_id).emplace_back(std::move(batch.at(dataset_id)));
//     }
//   }

//   return batches;
// }
}  // namespace thirdai::dataset