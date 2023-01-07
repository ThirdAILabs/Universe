#include "TabularDatasetLoader.h"
#include <dataset/src/DataSource.h>
#include <utils/Logging.h>

#include <utility>

namespace thirdai::dataset {

TabularDatasetLoader::TabularDatasetLoader(
    std::shared_ptr<dataset::DataSource> data_source,
    dataset::BatchProcessorPtr batch_processor, bool shuffle)
    : _data_source(std::move(data_source)),
      _batch_processor(std::move(batch_processor)),
      _shuffle(shuffle),
      _max_batch_size(_data_source->getMaxBatchSize()) {
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

std::optional<std::pair<InputDatasets, LabelDataset>>
TabularDatasetLoader::loadInMemory(uint64_t max_in_memory_batches) {
  std::vector<std::vector<BoltBatch>> batch_lists;

  uint64_t len = 0;
  uint64_t loaded_batches = 0;

  auto start = std::chrono::high_resolution_clock::now();

  while (auto batch_vector = nextBatchVector()) {
    len += batch_vector->at(0).getBatchSize();

    batch_lists.push_back(std::move(*batch_vector));

    loaded_batches++;
    if (loaded_batches >= max_in_memory_batches) {
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
  for (uint32_t dataset_id = 0; dataset_id < batch_lists.at(0).size();
       dataset_id++) {
    std::vector<BoltBatch> dataset_batches;
    dataset_batches.reserve(batch_lists.size());
    for (auto& batch_list : batch_lists) {
      dataset_batches.push_back(std::move(batch_list.at(dataset_id)));
    }
    dataset_ptrs.emplace_back(dataset_batches);
  }

  // For now assume labels is always the last dataset in the list
  // TODO(any): Once we have Bolt V2, fix this to work with an arbitrary
  // number of datasets and labels in arbitrary positions
  auto labels = dataset_ptrs.back();
  dataset_ptrs.pop_back();
  auto data = dataset_ptrs;

  return std::make_optional<std::pair<InputDatasets, LabelDataset>>(
      InputDatasets{data}, labels);
}

std::optional<std::vector<BoltBatch>> TabularDatasetLoader::nextBatchVector() {
  auto rows = _data_source->nextBatch();
  if (!rows) {
    return std::nullopt;
  }
  return _batch_processor->createBatch(*rows);
}

void TabularDatasetLoader::restart() {
  _data_source->restart();

  // When we restart we need to make sure we don't reread the header. s
  if (_batch_processor->expectsHeader()) {
    auto header = _data_source->nextLine();
    if (!header) {
      throw std::invalid_argument("Cannot read empty file.");
    }
  }
}

}  // namespace thirdai::dataset