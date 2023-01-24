#include "DatasetLoader.h"
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
      _shuffle(shuffle),
      _buffer_size(shuffle_config.buffer_size),
      _buffer(shuffle_config.seed),
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

// Loads the entire data source at once
std::pair<InputDatasets, LabelDataset> DatasetLoader::loadInMemory(
    size_t batch_size, bool verbose) {
  auto datasets =
      streamInMemory(/* batch_size = */ batch_size,
                     /* num_batches = */ std::numeric_limits<size_t>::max(),
                     /* verbose = */ verbose);
  if (!datasets) {
    throw std::invalid_argument(
        "Did not find any data to load from the data source.");
  }
  return datasets.value();
}

std::optional<std::pair<InputDatasets, LabelDataset>>
DatasetLoader::streamInMemory(size_t batch_size, size_t num_batches,
                              bool verbose) {
#if THIRDAI_EXPOSE_ALL
  if (verbose) {
    // This is useful internally but we don't want to expose it to keep the
    // output clear and simple.
    std::cout << "loading data | source '" << _data_source->resourceName()
              << "'" << std::endl;
  }
#endif

  auto start = std::chrono::high_resolution_clock::now();

  // TODO(Josh): Fix these calculations
  // We fill the buffer with num_batches + _batch_buffer_size number of batches
  // so that after exporting num_batches from the buffer we still have
  // _batch_buffer_size number of batches left for future shuffling.
  // We also much check if the sum overflows, since in the frequent case we
  // want to load all batches we pass in std::numeric_limits<size_t> which will
  // cause an overflow. For the source of this overflow check, see:
  // https://stackoverflow.com/q/199333/how-do-i-detect-unsigned-integer-overflow
  bool will_overflow =
      (std::numeric_limits<size_t>::max() - num_batches <= _buffer_size) ||
      (std::numeric_limits<size_t>::max() / batch_size <=
       num_batches + _buffer_size);
  size_t fill_size = will_overflow ? std::numeric_limits<size_t>::max()
                                   : (num_batches + _buffer_size) * batch_size;
  std::cout << "HERE " << fill_size << " " << num_batches << " " << _buffer_size << std::endl;
  fillShuffleBuffer(fill_size);

  auto batch_lists =
      _buffer.popBatches(num_batches, /* target_batch_size = */ batch_size);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

  if (batch_lists.at(0).empty()) {
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

  // For now assume labels is always the last dataset in the list
  // TODO(any): Once we have Bolt V2, fix this to work with an arbitrary
  // number of datasets and labels in arbitrary positions
  BoltDatasetPtr labels =
      std::make_shared<BoltDataset>(std::move(batch_lists.back()));
  std::vector<BoltDatasetPtr> data;
  for (uint32_t i = 0; i < batch_lists.size() - 1; i++) {
    data.push_back(std::make_shared<BoltDataset>(std::move(batch_lists.at(i))));
  }

  if (verbose) {
    std::cout << "loaded data | source '" << _data_source->resourceName()
              << "' | vectors " << labels->len() << " | batches "
              << labels->numBatches() << " | time " << duration
              << "s | complete\n"
              << std::endl;
  }
  return std::make_pair(data, labels);
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

  _buffer.clear();
}

void DatasetLoader::fillShuffleBuffer(size_t num_rows) {
  while (_buffer.size() <= num_rows) {
    auto rows = _data_source->nextBatch(
        /* target_batch_size = */ _featurization_batch_size);
    if (!rows) {
      return;
    }

    auto batch = _featurizer->createBatch(*rows);
    _buffer.insertBatch(std::move(batch), _shuffle);
  }
}

}  // namespace thirdai::dataset