#include "DatasetLoader.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/Datasets.h>
#include <utils/Logging.h>
#include <limits>
#include <optional>
#include <utility>

namespace thirdai::dataset {

DatasetLoader::DatasetLoader(DataSourcePtr data_source,
                             dataset::FeaturizerPtr featurizer, bool shuffle,
                             uint32_t shuffle_seed,
                             size_t internal_featurization_batch_size)
    : _data_source(std::move(data_source)),
      _featurizer(std::move(featurizer)),
      _shuffler(/* shuffle= */ shuffle, /* seed= */ shuffle_seed),
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

  bool will_overflow =
      num_batches > (std::numeric_limits<size_t>::max() / batch_size);
  size_t fill_size = will_overflow ? std::numeric_limits<size_t>::max()
                                   : num_batches * batch_size;
  fillShuffler(fill_size);
  auto data = _shuffler.datasets(/* batch_size= */ batch_size);
  if (!data) {
    return std::nullopt;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

  if (data->at(0)->len() == 0) {
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

  if (verbose) {
    std::cout << "loaded data | source '" << _data_source->resourceName()
              << "' | vectors " << data->at(0)->len() << " | batches "
              << data->at(0)->numBatches() << " | time " << duration
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

void DatasetLoader::fillShuffler(size_t num_rows) {
  while (_shuffler.size() <= num_rows) {
    auto rows = _data_source->nextBatch(
        /* target_batch_size = */ _featurization_batch_size);
    if (!rows) {
      return;
    }
    auto vectors = _featurizer->featurize(*rows);
    std::vector<BoltBatch> batch;
    batch.reserve(vectors.size());
    for (auto& vector_list : vectors) {
      batch.emplace_back(std::move(vector_list));
    }
    _shuffler.add(std::move(batch));
  }
}

}  // namespace thirdai::dataset