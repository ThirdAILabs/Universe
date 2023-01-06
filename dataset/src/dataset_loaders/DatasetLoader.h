#include <dataset/src/Datasets.h>

namespace thirdai::dataset {

using InputDatasets = std::vector<dataset::BoltDatasetPtr>;
using LabelDataset = dataset::BoltDatasetPtr;

class DatasetLoader {
 public:
  virtual std::optional<std::pair<InputDatasets, LabelDataset>> loadInMemory(
      uint32_t max_in_memory_batches) = 0;

  virtual void restart() = 0;

  virtual ~DatasetLoader() = default;
};

using DatasetLoaderPtr = std::unique_ptr<DatasetLoader>;

} // namespace thirdai::dataset