#pragma once

#include <dataset/src/Datasets.h>
#include <dataset/src/VectorBuffer.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <stdexcept>

namespace thirdai::dataset {

struct DatasetShuffleConfig {
  DatasetShuffleConfig() : min_buffer_size(64000), seed(time(NULL)) {}

  explicit DatasetShuffleConfig(size_t min_vecs_in_buffer)
      : min_buffer_size(min_vecs_in_buffer), seed(time(NULL)) {}

  DatasetShuffleConfig(size_t min_vecs_in_buffer, uint32_t seed)
      : min_buffer_size(min_vecs_in_buffer), seed(seed) {}

  size_t min_buffer_size;
  uint32_t seed;
};

const uint32_t DEFAULT_FEATURIZATION_BATCH_SIZE = 2048;

using InputDatasets = std::vector<dataset::BoltDatasetPtr>;
using LabelDataset = dataset::BoltDatasetPtr;
using DatasetSlice = std::vector<BoltBatch>;
class DatasetLoader final {
 public:
  DatasetLoader(std::shared_ptr<dataset::DataSource> data_source,
                dataset::FeaturizerPtr featurizer, bool shuffle,
                DatasetShuffleConfig shuffle_config = DatasetShuffleConfig(),
                size_t internal_featurization_batch_size =
                    DEFAULT_FEATURIZATION_BATCH_SIZE);

  // TODO(Josh/Geordie/Nick/David): We should generalize these next two load
  // methods to return a vector of BoltDatasets, and figure out which are
  // inputs and which are labels in UDT

  std::pair<InputDatasets, LabelDataset> loadAll(size_t batch_size,
                                                 bool verbose = true);

  std::optional<std::pair<InputDatasets, LabelDataset>> loadSome(
      size_t batch_size, size_t num_batches, bool verbose = true);

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
  void fillVectorBuffer(size_t num_rows);

  // Pops _featurizer.numDatasets() DatasetSlices from _buffer. The ith
  // DatasetSlice corresponds to the ith BoltDataset this DatasetLoader is
  // loading (out of _featurizer.numDatasets() total datasets). Each
  // DatasetSlice will have the same number of batches and the same batch sizes.
  std::vector<DatasetSlice> popFromBuffer(size_t target_num_batches,
                                          size_t target_batch_size);

  DataSourcePtr _data_source;
  std::shared_ptr<Featurizer> _featurizer;

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