#include "DataLoader.h"
#include "StreamingDataset.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/bolt_datasets/ShuffleableStreamingDataset.h>
#include <dataset/src/bolt_datasets/batch_processors/GenericBatchProcessor.h>
#include <memory>

namespace thirdai::dataset {

class StreamingGenericDatasetLoader {
 public:
  StreamingGenericDatasetLoader(
      std::string filename, std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks, uint32_t batch_size,
      bool has_header = false, char delimiter = ',')
      : _processor(std::make_shared<GenericBatchProcessor>(
            std::move(input_blocks), std::move(label_blocks), has_header,
            delimiter)),
        _streamer(std::make_shared<SimpleFileDataLoader>(filename, batch_size),
                  _processor) {}

  std::optional<BoltDataLabelPair<bolt::BoltBatch>> nextBatch() {
    return _streamer.nextBatch();
  }

  std::pair<std::shared_ptr<InMemoryDataset<bolt::BoltBatch>>, BoltDatasetPtr>
  loadInMemory() {
    return _streamer.loadInMemory();
  }

  uint32_t getMaxBatchSize() const { return _streamer.getMaxBatchSize(); }

  uint32_t getInputDim() { return _processor->getInputDim(); }

  uint32_t getLabelDim() { return _processor->getLabelDim(); }

 private:
  std::shared_ptr<GenericBatchProcessor> _processor;
  ShuffleableStreamingDataset _streamer;
};

}  // namespace thirdai::dataset