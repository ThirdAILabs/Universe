#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/VectorBuffer.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <utils/Random.h>
#include <stdexcept>

namespace thirdai::dataset {

struct DatasetShuffleConfig {
  explicit DatasetShuffleConfig(size_t min_vecs_in_buffer = 64000,
                                uint32_t seed = global_random::nextSeed())
      : min_buffer_size(min_vecs_in_buffer), seed(seed) {}

  size_t min_buffer_size;
  uint32_t seed;
};

const uint32_t DEFAULT_FEATURIZATION_BATCH_SIZE = 2048;

using DatasetSlice = std::vector<BoltBatch>;
class DatasetLoader final {
 public:
  DatasetLoader(std::shared_ptr<dataset::DataSource> data_source,
                dataset::FeaturizerPtr featurizer, bool shuffle,
                DatasetShuffleConfig shuffle_config = DatasetShuffleConfig(),
                size_t internal_featurization_batch_size =
                    DEFAULT_FEATURIZATION_BATCH_SIZE,
                size_t data_source_batch_to_skip = 0);

  std::vector<BoltDatasetPtr> loadAll(size_t batch_size, bool verbose = true);

  std::optional<std::vector<BoltDatasetPtr>> loadSome(size_t batch_size,
                                                      size_t num_batches,
                                                      bool verbose = true);

  size_t currentDataSourceBatch() const { return _current_data_source_batch; }

  void restart();

  uint32_t getInputDim() {
    // TODO(Nick/Geordie): Replace this with a getInputDims() call.
    return _featurizer->getDimensions().at(0);
  }

  uint32_t getLabelDim() { return _featurizer->getDimensions().at(1); }

  void manuallyAddToBuffer(std::vector<BoltVector>&& vectors);

 private:
  // Adds batches to the buffer until the data source is finished or the buffer
  // reaches the passed in number of rows
  void fillVectorBuffer(size_t num_rows);

  // Pops _featurizer.numDatasets() DatasetSlices from _buffer. The ith
  // DatasetSlice corresponds to the ith BoltDataset this DatasetLoader is
  // loading (out of _featurizer.numDatasets() total datasets). Each
  // DatasetSlice will have the same number of batches and the same batch sizes.
  std::vector<DatasetSlice> popFromBuffer(size_t target_num_batches,
                                          size_t target_batch_size);

  DataSourcePtr _data_source;
  std::shared_ptr<Featurizer> _featurizer;
  std::optional<std::string> _header = std::nullopt;
  size_t _current_data_source_batch;

  bool _shuffle;
  // We try to ensure at least this many batches are in the buffer and shuffled
  // when we return shuffled values
  size_t _buffer_size;
  VectorBuffer _buffer;

  // Batch size we use for loading from the data source and passing to the
  // Featurizer
  size_t _featurization_batch_size;
};

using DatasetLoaderPtr = std::unique_ptr<DatasetLoader>;

}  // namespace thirdai::dataset