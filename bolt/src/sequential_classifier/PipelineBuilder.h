#pragma once

#include "Schema.h"
#include <dataset/src/bolt_datasets/StreamingGenericDatasetLoader.h>
#include <dataset/src/encodings/categorical/StringToUidMap.h>
#include <dataset/src/encodings/count_history/CountHistoryIndex.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

const size_t BATCH_SIZE = 2048;
using Blocks = std::vector<std::shared_ptr<dataset::Block>>;

struct PersistentPipelineStates {
  std::shared_ptr<dataset::StringToUidMap> item_id_map;
  std::shared_ptr<dataset::StringToUidMap> target_id_map;
  std::vector<std::shared_ptr<dataset::StringToUidMap>> cat_attr_maps;
  std::vector<std::shared_ptr<dataset::CountHistoryIndex>> trackable_counts;
};

class PipelineBuilder {
 public:
  PipelineBuilder(Schema schema, char delimiter);

  std::shared_ptr<dataset::StreamingGenericDatasetLoader> buildPipelineForFile(
      std::string& filename, bool shuffle, bool overwrite_index);

  PersistentPipelineStates _states;

 private:
  static std::string getHeader(dataset::DataLoader& loader);

  Blocks buildInputBlocks(bool overwrite_index);

  Blocks buildLabelBlocks();

  size_t autotuneShuffleBufferSize() const;

  void addDateFeats(Blocks& blocks);

  void addItemIdFeats(Blocks& blocks);

  void addTextAttrFeats(Blocks& blocks);

  void addCategoricalAttrFeats(Blocks& blocks);

  void addTrackableQtyFeats(Blocks& blocks, bool overwrite_index);

  void addNonzeros(size_t nonzeros) { _est_nonzeros += nonzeros; }

  Schema _schema;
  size_t _est_nonzeros;
  char _delimiter;
};

}  // namespace thirdai::bolt