#pragma once

#include <dataset/src/Datasets.h>

namespace thirdai::dataset {

using InputDatasets = std::vector<dataset::BoltDatasetPtr>;
using LabelDataset = dataset::BoltDatasetPtr;

class DatasetLoader {
 public:
  virtual std::optional<std::pair<InputDatasets, LabelDataset>> loadInMemory(
      uint64_t max_in_memory_batches) = 0;

  // TODO(Josh): Does this need to be virtual
  std::pair<InputDatasets, LabelDataset> loadInMemory() {
    auto datasets = loadInMemory(std::numeric_limits<uint64_t>::max());
    if (!datasets) {
      throw std::invalid_argument(
          "Did not find any data to load from the data source.");
    }
    return datasets.value();
  }

  virtual void restart() = 0;

  virtual ~DatasetLoader() = default;
};

using DatasetLoaderPtr = std::unique_ptr<DatasetLoader>;

}  // namespace thirdai::dataset