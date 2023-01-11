#include "DatasetLoader.h"
#include <dataset/src/DataSource.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/ShuffleBatchBuffer.h>
#include <utils/Logging.h>
#include <utility>

namespace thirdai::dataset {

DatasetLoader::DatasetLoader(DataSourcePtr data_source,
                             dataset::BatchProcessorPtr batch_processor,
                             bool shuffle, DatasetShuffleConfig shuffle_config)
    : _data_source(std::move(data_source)),
      _batch_processor(std::move(batch_processor)),
      _shuffle(shuffle),
      _batch_buffer_size(shuffle_config.n_batches),
      _buffer(shuffle_config.seed, _data_source->getMaxBatchSize()) {
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

// Loads the entire data source at once
std::pair<InputDatasets, LabelDataset> DatasetLoader::loadInMemory() {
  auto datasets = loadInMemory(std::numeric_limits<uint64_t>::max());
  if (!datasets) {
    throw std::invalid_argument(
        "Did not find any data to load from the data source.");
  }
  return datasets.value();
}

std::optional<std::pair<InputDatasets, LabelDataset>>
DatasetLoader::loadInMemory(uint64_t num_batches) {
#if THIRDAI_EXPOSE_ALL
  // This is useful internally but we don't want to expose it to keep the
  // output clear and simple.
  std::cout << "loading data | source '" << _data_source->resourceName() << "'"
            << std::endl;
#endif

  auto start = std::chrono::high_resolution_clock::now();

  fillShuffleBuffer(
      /* fill_size = */ std::max<size_t>(num_batches, _batch_buffer_size));

  auto batch_lists = _buffer.exportBuffer(num_batches);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

  if (batch_lists.at(0).empty()) {
#if THIRDAI_EXPOSE_ALL
    // This is to ensure that it always prints complete if it prints that it
    // has started loading above.
    std::cout << "loading data | source '" << _data_source->resourceName()
              << "' | vectors 0 | batches 0 | time " << duration
              << "s | complete\n"
              << std::endl;
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

  std::cout << "loading data | source '" << _data_source->resourceName()
            << "' | vectors " << labels->len() << " | batches "
            << labels->numBatches() << " | time " << duration
            << "s | complete\n"
            << std::endl;

  return std::make_pair(data, labels);
}

void DatasetLoader::restart() {
  _data_source->restart();

  // When we restart we need to make sure we don't reread the header. s
  if (_batch_processor->expectsHeader()) {
    auto header = _data_source->nextLine();
    if (!header) {
      throw std::invalid_argument("Cannot read empty file.");
    }
  }

  _buffer.clear();
}

void DatasetLoader::fillShuffleBuffer(size_t fill_size) {
  while (_buffer.size() <= fill_size) {
    auto rows = _data_source->nextBatch();
    if (!rows) {
      return;
    }

    auto batch = _batch_processor->createBatch(*rows);
    _buffer.insertBatch(std::move(batch), _shuffle);
  }
}

}  // namespace thirdai::dataset