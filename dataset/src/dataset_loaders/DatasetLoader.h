#pragma once

#include <dataset/src/Datasets.h>
#include <dataset/src/Shuffler.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <stdexcept>

namespace thirdai::dataset {

const uint32_t DEFAULT_FEATURIZATION_BATCH_SIZE = 2048;

using DatasetSlice = std::vector<BoltBatch>;
class DatasetLoader final {
 public:
  DatasetLoader(std::shared_ptr<dataset::DataSource> data_source,
                dataset::FeaturizerPtr featurizer, bool shuffle,
                uint32_t shuffle_seed = time(NULL),
                size_t internal_featurization_batch_size =
                    DEFAULT_FEATURIZATION_BATCH_SIZE);

  std::vector<BoltDatasetPtr> loadAll(size_t batch_size, bool verbose = true);

  std::optional<std::vector<BoltDatasetPtr>> loadSome(size_t batch_size,
                                                      size_t num_batches,
                                                      bool verbose = true);

  void restart();

  uint32_t getInputDim() {
    // TODO(Josh): This is assuming we have one input and one label
    // dataset
    return _featurizer->getDimensions().at(0);
  }

  uint32_t getLabelDim() {
    // TODO(Josh): Again, this is assuming we have one input and one label
    // dataset
    return _featurizer->getDimensions().at(1);
  }

 private:
  // Adds batches to the buffer until the data source is finished or the buffer
  // reaches the passed in number of rows
  void fillShuffler(size_t num_rows);

  DataSourcePtr _data_source;
  std::shared_ptr<Featurizer> _featurizer;

  Shuffler _shuffler;

  // Batch size we use for loading from the data source and passing to the
  // Featurizer
  size_t _featurization_batch_size;
};

using DatasetLoaderPtr = std::unique_ptr<DatasetLoader>;

}  // namespace thirdai::dataset