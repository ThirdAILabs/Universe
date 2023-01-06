#include "TabularDatasetLoader.h"

namespace thirdai::dataset {

TabularDatasetLoader::TabularDatasetLoader(
    std::shared_ptr<dataset::DataLoader> data_loader,
    dataset::GenericBatchProcessorPtr batch_processor, bool shuffle)
    : _dataset(std::move(data_loader), std::move(batch_processor), shuffle) {}

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

} // namespace thirdai::dataset