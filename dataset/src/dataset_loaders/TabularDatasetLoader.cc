#include "TabularDatasetLoader.h"
#include <dataset/src/DataSource.h>

namespace thirdai::dataset {

TabularDatasetLoader::TabularDatasetLoader(
    const std::shared_ptr<dataset::DataSource>& data_source,
    dataset::GenericBatchProcessorPtr batch_processor, bool shuffle)
    : _dataset(data_source, std::move(batch_processor), shuffle) {}

std::optional<std::pair<InputDatasets, LabelDataset>>
TabularDatasetLoader::loadInMemory(uint32_t max_in_memory_batches) {
  auto datasets = _dataset.loadInMemoryWithMaxBatches(max_in_memory_batches);
  if (!datasets) {
    return std::nullopt;
  }

  auto& [data, labels] = datasets.value();

  return std::make_optional<std::pair<InputDatasets, LabelDataset>>(
      InputDatasets{data}, labels);
}

}  // namespace thirdai::dataset