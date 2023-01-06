#include "DatasetLoader.h"
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>

namespace thirdai::dataset {

class TabularDatasetLoader final : public DatasetLoader {
 public:
  TabularDatasetLoader(std::shared_ptr<dataset::DataLoader> data_loader,
                       dataset::GenericBatchProcessorPtr batch_processor,
                       bool shuffle);

  std::optional<std::pair<InputDatasets, LabelDataset>> loadInMemory(
      uint32_t max_in_memory_batches) final;

  void restart() final { _dataset.restart(); }

 private:
  dataset::StreamingGenericDatasetLoader _dataset;
};

using TabularDatasetLoaderPtr = std::unique_ptr<TabularDatasetLoader>;

} // namespace thirdai::dataset