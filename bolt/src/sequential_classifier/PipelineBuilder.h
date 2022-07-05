#pragma once

#include "SchemaProcessor.h"
#include <dataset/src/bolt_datasets/StreamingGenericDatasetLoader.h>
#include <dataset/src/encodings/categorical/StringToUidMap.h>
#include <dataset/src/encodings/count_history/CountHistoryIndex.h>
#include <memory>
#include <unordered_map>

namespace thirdai::bolt {

const size_t BATCH_SIZE = 2048;
using Blocks = std::vector<std::shared_ptr<dataset::Block>>;

struct SequentialClassifierConfig {
  SequentialClassifierConfig(std::string model_size, size_t n_target_classes,
                             size_t horizon, size_t n_items, size_t n_users = 0,
                             size_t n_item_categories = 0,
                             dataset::GraphPtr user_graph = nullptr,
                             size_t user_max_neighbors = 0,
                             dataset::GraphPtr item_graph = nullptr,
                             size_t item_max_neighbors = 0)
      : _n_users(n_users),
        _n_items(n_items),
        _n_categories(n_item_categories),
        _horizon(horizon),
        _n_target_classes(n_target_classes),
        _model_size(std::move(model_size)),
        _item_graph(std::move(item_graph)),
        _item_max_neighbors(item_max_neighbors),
        _user_graph(std::move(user_graph)),
        _user_max_neighbors(user_max_neighbors) {}

  size_t _n_users;
  size_t _n_items;
  size_t _n_categories;
  size_t _horizon;
  size_t _n_target_classes;
  std::string _model_size;
  dataset::GraphPtr _item_graph;
  size_t _item_max_neighbors;
  dataset::GraphPtr _user_graph;
  size_t _user_max_neighbors;
};

struct PersistentPipelineStates {
  std::shared_ptr<dataset::CountHistoryIndex> _user_history;
  std::shared_ptr<dataset::CountHistoryIndex> _item_history;
  std::shared_ptr<dataset::StringToUidMap> _user_id_map;
  std::shared_ptr<dataset::StringToUidMap> _item_id_map;
  std::shared_ptr<dataset::StringToUidMap> _target_id_map;
  std::shared_ptr<dataset::StringToUidMap> _cat_attr_map;
};

class PipelineBuilder {
 public:
  PipelineBuilder(GivenSchema& schema, SequentialClassifierConfig& config,
                  char delimiter);

  std::shared_ptr<dataset::StreamingGenericDatasetLoader> buildPipelineForFile(
      std::string& filename, bool shuffle, bool overwrite_index);

  PersistentPipelineStates _states;

 private:
  static std::string getHeader(dataset::DataLoader& loader);

  Blocks buildInputBlocks(const ColumnNumbers& columns, bool overwrite_index);

  Blocks buildLabelBlocks(const ColumnNumbers& columns);

  size_t autotuneShuffleBufferSize() const;

  void addDateFeats(const ColumnNumbers& columns, Blocks& blocks);

  void addUserIdFeats(const ColumnNumbers& columns, Blocks& blocks);

  void addItemIdFeats(const ColumnNumbers& columns, Blocks& blocks);

  void addTextAttrFeats(const ColumnNumbers& columns, Blocks& blocks);

  void addCategoricalAttrFeats(const ColumnNumbers& columns, Blocks& blocks);

  void addUserTemporalFeats(const ColumnNumbers& columns, Blocks& blocks,
                            bool overwrite_index);

  void addItemTemporalFeats(const ColumnNumbers& columns, Blocks& blocks,
                            bool overwrite_index);

  void addNonzeros(size_t nonzeros) { _est_nonzeros += nonzeros; }

  void checkCategoricalMap(std::shared_ptr<dataset::StringToUidMap>& map);

  static void checkCountHistoryIndex(
      std::shared_ptr<dataset::CountHistoryIndex>& index, bool overwrite_index);

  static std::pair<bool, size_t> countCol(const ColumnNumbers& columns);

  uint32_t lookback() const {
    return std::max(_config._horizon, static_cast<size_t>(30));
  }

  SchemaProcessor _schema_processor;
  SequentialClassifierConfig _config;
  size_t _est_nonzeros;
  char _delimiter;
};

}  // namespace thirdai::bolt