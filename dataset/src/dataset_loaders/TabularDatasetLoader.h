#include "DatasetLoader.h"
#include <dataset/src/Datasets.h>

namespace thirdai::dataset {

class TabularDatasetLoader final : public DatasetLoader {
 public:
  TabularDatasetLoader(std::shared_ptr<dataset::DataSource> data_source,
                      dataset::BatchProcessorPtr batch_processor,
                       bool shuffle);

  std::optional<std::pair<InputDatasets, LabelDataset>> loadInMemory(
      uint64_t max_in_memory_batches) final;

  void restart() final;

 private:
  std::optional<std::vector<BoltBatch>> nextBatchVector();

  std::shared_ptr<DataSource> _data_source;
  std::shared_ptr<BatchProcessor> _batch_processor;
  bool _shuffle;
  uint32_t _max_batch_size;
};

using TabularDatasetLoaderPtr = std::unique_ptr<TabularDatasetLoader>;

}  // namespace thirdai::dataset