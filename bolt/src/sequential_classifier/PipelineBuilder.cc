#include <bolt/src/sequential_classifier/PipelineBuilder.h>
#include <bolt/src/utils/AutoTuneUtils.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/blocks/Trend.h>

namespace thirdai::bolt {

PipelineBuilder::PipelineBuilder(GivenSchema& schema,
                                 SequentialClassifierConfig& config,
                                 const char delimiter)
    : _schema_processor(schema), _config(config), _delimiter(delimiter) {}

std::shared_ptr<dataset::StreamingGenericDatasetLoader>
PipelineBuilder::buildPipelineForFile(std::string& filename, bool shuffle,
                                      bool overwrite_index) {
  _est_nonzeros = 0;

  auto loader =
      std::make_shared<dataset::SimpleFileDataLoader>(filename, BATCH_SIZE);
  auto header = getHeader(*loader);
  auto columns = _schema_processor.parseHeader(header, _delimiter);

  auto input_blocks = buildInputBlocks(columns, overwrite_index);
  auto label_blocks = buildLabelBlocks(columns);
  auto buffer_size = autotuneShuffleBufferSize();

  return std::make_shared<dataset::StreamingGenericDatasetLoader>(
      loader, input_blocks, label_blocks, shuffle,
      dataset::ShuffleBufferConfig(buffer_size),
      /* has_header = */ false, _delimiter);
}

std::string PipelineBuilder::getHeader(dataset::DataLoader& loader) {
  auto header = loader.getHeader();
  if (!header) {
    throw std::invalid_argument(
        "[SequentialClassifier::train] The file has no header.");
  }
  return *header;
}

Blocks PipelineBuilder::buildInputBlocks(const ColumnNumbers& columns,
                                         bool overwrite_index) {
  std::vector<std::shared_ptr<dataset::Block>> blocks;
  addDateFeats(columns, blocks);
  addUserIdFeats(columns, blocks);
  addItemIdFeats(columns, blocks);
  addTextAttrFeats(columns, blocks);
  addCategoricalAttrFeats(columns, blocks);
  addUserTemporalFeats(columns, blocks, overwrite_index);
  addItemTemporalFeats(columns, blocks, overwrite_index);

  return blocks;
}

Blocks PipelineBuilder::buildLabelBlocks(const ColumnNumbers& columns) {
  checkCategoricalMap(_states._target_id_map);
  return {std::make_shared<dataset::CategoricalBlock>(
      columns.at(SchemaKey::target), _states._target_id_map)};
}

size_t PipelineBuilder::autotuneShuffleBufferSize() const {
  auto batch_mem =
      BATCH_SIZE * _est_nonzeros * 8;  // 4 bytes for index, 4 bytes for value.
  if (auto ram = AutoTuneUtils::getSystemRam()) {
    auto mem_allowance = *ram / 2;
    return mem_allowance / batch_mem;
  }
  return 1000;
}

void PipelineBuilder::addDateFeats(const ColumnNumbers& columns,
                                   Blocks& blocks) {
  blocks.push_back(
      std::make_shared<dataset::DateBlock>(columns.at(SchemaKey::timestamp)));
  addNonzeros(4);
}

void PipelineBuilder::addUserIdFeats(const ColumnNumbers& columns,
                                     Blocks& blocks) {
  if (columns.count(SchemaKey::user) == 0) {
    return;
  }
  checkCategoricalMap(_states._user_id_map);
  blocks.push_back(std::make_shared<dataset::CategoricalBlock>(
      columns.at(SchemaKey::user), _states._user_id_map, _config._user_graph,
      _config._user_max_neighbors));
  addNonzeros(1);
}

void PipelineBuilder::addItemIdFeats(const ColumnNumbers& columns,
                                     Blocks& blocks) {
  checkCategoricalMap(_states._item_id_map);
  blocks.push_back(std::make_shared<dataset::CategoricalBlock>(
      columns.at(SchemaKey::user), _states._item_id_map, _config._item_graph,
      _config._item_max_neighbors));
  addNonzeros(1);
}

void PipelineBuilder::addTextAttrFeats(const ColumnNumbers& columns,
                                       Blocks& blocks) {
  if (columns.count(SchemaKey::text_attr) == 0) {
    return;
  }
  blocks.push_back(std::make_shared<dataset::TextBlock>(
      columns.at(SchemaKey::text_attr), /* dim = */ 100000));
  addNonzeros(100);
}

void PipelineBuilder::addCategoricalAttrFeats(const ColumnNumbers& columns,
                                              Blocks& blocks) {
  if (columns.count(SchemaKey::categorical_attr) == 0) {
    return;
  }
  checkCategoricalMap(_states._cat_attr_map);
  blocks.push_back(std::make_shared<dataset::CategoricalBlock>(
      columns.at(SchemaKey::categorical_attr), _states._cat_attr_map));
  addNonzeros(1);
}

void PipelineBuilder::addUserTemporalFeats(const ColumnNumbers& columns,
                                           Blocks& blocks,
                                           bool overwrite_index) {
  if (columns.count(SchemaKey::user) == 0) {
    return;
  }

  checkCountHistoryIndex(_states._user_history, overwrite_index);

  auto [has_count_col, count_col] = countCol(columns);
  auto trend_block = std::make_shared<dataset::TrendBlock>(
      has_count_col, columns.at(SchemaKey::user),
      columns.at(SchemaKey::timestamp), count_col, _config._horizon, lookback(),
      _states._user_history);

  blocks.push_back(trend_block);
  addNonzeros(trend_block->featureDim());
}

void PipelineBuilder::addItemTemporalFeats(const ColumnNumbers& columns,
                                           Blocks& blocks,
                                           bool overwrite_index) {
  checkCountHistoryIndex(_states._item_history, overwrite_index);

  auto [has_count_col, count_col] = countCol(columns);
  auto trend_block = std::make_shared<dataset::TrendBlock>(
      has_count_col, columns.at(SchemaKey::item),
      columns.at(SchemaKey::timestamp), count_col, _config._horizon, lookback(),
      _states._item_history);

  blocks.push_back(trend_block);
  addNonzeros(trend_block->featureDim());
}

void PipelineBuilder::checkCategoricalMap(
    std::shared_ptr<dataset::StringToUidMap>& map) {
  if (map == nullptr) {
    map = std::make_shared<dataset::StringToUidMap>(_config._n_users);
  }
}

void PipelineBuilder::checkCountHistoryIndex(
    std::shared_ptr<dataset::CountHistoryIndex>& index, bool overwrite_index) {
  if (index == nullptr || overwrite_index) {
    index = std::make_shared<dataset::CountHistoryIndex>(
        /* n_rows = */ 5, /* range_pow = */ 22,
        /* lifetime = */ std::numeric_limits<uint32_t>::max());
  }
}

std::pair<bool, size_t> PipelineBuilder::countCol(
    const ColumnNumbers& columns) {
  bool has_count_col = columns.count(SchemaKey::trackable_quantity) != 0;
  size_t count_col =
      has_count_col ? columns.at(SchemaKey::trackable_quantity) : 0;
  return {has_count_col, count_col};
}

}  // namespace thirdai::bolt