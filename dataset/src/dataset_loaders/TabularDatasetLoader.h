#include "DatasetLoader.h"
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>

namespace thirdai::dataset {

class TabularDatasetLoader final : public DatasetLoader {
 public:
  TabularDatasetLoader(const std::shared_ptr<dataset::DataSource>& data_source,
                       dataset::GenericBatchProcessorPtr batch_processor,
                       bool shuffle);

  std::optional<std::pair<InputDatasets, LabelDataset>> loadInMemory(
      uint32_t max_in_memory_batches) final;

  void restart() final { _dataset.restart(); }

 private:
  dataset::StreamingGenericDatasetLoader _dataset;
};

using TabularDatasetLoaderPtr = std::unique_ptr<TabularDatasetLoader>;

}  // namespace thirdai::dataset