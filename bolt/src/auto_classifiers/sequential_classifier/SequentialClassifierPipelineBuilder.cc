#include <bolt/src/auto_classifiers/AutoClassifierUtils.h>
#include <bolt/src/auto_classifiers/sequential_classifier/SequentialClassifierPipelineBuilder.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/CategoricalTracking.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/blocks/Trend.h>
#include <dataset/src/encodings/categorical/StringToUidMap.h>
#include <dataset/src/encodings/categorical_history/CategoricalHistoryIndex.h>
#include <dataset/src/encodings/count_history/CountHistoryIndex.h>
#include <memory>
#include <vector>

namespace thirdai::bolt {

SequentialClassifierPipelineBuilder::SequentialClassifierPipelineBuilder(
    SequentialClassifierSchema schema, const char delimiter)
    : _schema(std::move(schema)), _delimiter(delimiter) {}

std::shared_ptr<dataset::StreamingGenericDatasetLoader>
SequentialClassifierPipelineBuilder::buildPipelineForFile(
    const std::string& filename, bool shuffle, bool overwrite_index) {
  _est_nonzeros = 0;

  auto loader =
      std::make_shared<dataset::SimpleFileDataLoader>(filename, BATCH_SIZE);
  auto header = getHeader(*loader);
  _schema.fitToHeader(header, _delimiter);

  auto input_blocks = buildInputBlocks(overwrite_index);
  auto label_blocks = buildLabelBlocks();
  auto buffer_size = autotuneShuffleBufferSize();

  return std::make_shared<dataset::StreamingGenericDatasetLoader>(
      loader, input_blocks, label_blocks, shuffle,
      dataset::ShuffleBufferConfig(buffer_size),
      /* has_header = */ false, _delimiter);
}

std::string SequentialClassifierPipelineBuilder::getHeader(
    dataset::DataLoader& loader) {
  auto header = loader.getHeader();
  if (!header) {
    throw std::invalid_argument("The file has no header.");
  }
  return *header;
}

Blocks SequentialClassifierPipelineBuilder::buildInputBlocks(
    bool overwrite_index) {
  std::vector<std::shared_ptr<dataset::Block>> blocks;
  addDateFeats(blocks);
  addItemIdFeats(blocks);
  addTextAttrFeats(blocks);
  addCategoricalAttrFeats(blocks);
  addTrackableQtyFeats(blocks, overwrite_index);
  addTrackableCatFeats(blocks);  // must be called after item id feats.
  return blocks;
}

Blocks SequentialClassifierPipelineBuilder::buildLabelBlocks() {
  auto target = _schema.target;
  if (_states.target_id_map == nullptr) {
    _states.target_id_map =
        std::make_shared<dataset::StringToUidMap>(target.n_distinct);
  }

  return {std::make_shared<dataset::CategoricalBlock>(target.col_num,
                                                      _states.target_id_map)};
}

size_t SequentialClassifierPipelineBuilder::autotuneShuffleBufferSize() const {
  auto batch_mem =
      BATCH_SIZE * _est_nonzeros * 8;  // 4 bytes for index, 4 bytes for value.
  if (auto ram = AutoClassifierUtils::getSystemRam()) {
    auto mem_allowance = *ram / 2;
    return mem_allowance / batch_mem;
  }
  return 1000;
}

void SequentialClassifierPipelineBuilder::addDateFeats(Blocks& blocks) {
  blocks.push_back(
      std::make_shared<dataset::DateBlock>(_schema.timestamp.col_num));
  addNonzeros(4);
}

void SequentialClassifierPipelineBuilder::addItemIdFeats(Blocks& blocks) {
  auto item = _schema.item;
  if (_states.item_id_map == nullptr) {
    _states.item_id_map =
        std::make_shared<dataset::StringToUidMap>(item.n_distinct);
  }

  blocks.push_back(std::make_shared<dataset::CategoricalBlock>(
      item.col_num, _states.item_id_map, item.graph, item.max_neighbors));
  addNonzeros(1);
}

void SequentialClassifierPipelineBuilder::addTextAttrFeats(Blocks& blocks) {
  auto text_attrs = _schema.text_attributes;
  for (const auto& text : text_attrs) {
    blocks.push_back(
        std::make_shared<dataset::TextBlock>(text.col_num, /* dim = */ 100000));
    addNonzeros(20);
  }
}

void SequentialClassifierPipelineBuilder::addCategoricalAttrFeats(
    Blocks& blocks) {
  auto cat_attrs = _schema.categorical_attributes;
  for (uint32_t i = 0; i < cat_attrs.size(); i++) {
    auto cat = cat_attrs[i];
    if (i >= _states.cat_attr_maps.size()) {
      _states.cat_attr_maps.push_back(
          std::make_shared<dataset::StringToUidMap>(cat.n_distinct));
    }
    blocks.push_back(std::make_shared<dataset::CategoricalBlock>(
        cat.col_num, _states.cat_attr_maps[i]));
    addNonzeros(1);
  }
}

void SequentialClassifierPipelineBuilder::addTrackableQtyFeats(
    Blocks& blocks, bool overwrite_index) {
  if (overwrite_index) {
    _states.trackable_counts =
        std::vector<std::shared_ptr<dataset::CountHistoryIndex>>();
  }

  auto item = _schema.item;
  auto config = _schema.tracking_config;
  auto trackable_qty = _schema.trackable_quantities;

  for (uint32_t i = 0; i < trackable_qty.size(); i++) {
    auto qty = trackable_qty[i];
    if (i >= _states.trackable_counts.size()) {
      _states.trackable_counts.push_back(
          std::make_shared<dataset::CountHistoryIndex>(
              /* n_rows = */ 5, /* range_pow = */ 22,
              /* lifetime = */ std::numeric_limits<uint32_t>::max()));
    }
    auto trend_block = std::make_shared<dataset::TrendBlock>(
        qty.has_col_num, item.col_num, _schema.timestamp.col_num, qty.col_num,
        config.lookahead, config.lookback, config.period,
        _states.trackable_counts[i], item.graph, item.max_neighbors);
    blocks.push_back(trend_block);
    addNonzeros(trend_block->featureDim());
  }
}

void SequentialClassifierPipelineBuilder::addTrackableCatFeats(Blocks& blocks) {
  auto item = _schema.item;
  auto config = _schema.tracking_config;
  auto trackable_cats = _schema.trackable_categories;

  for (uint32_t i = 0; i < trackable_cats.size(); i++) {
    auto cat = trackable_cats[i];
    if (i >= _states.trackable_categories.size()) {
      _states.trackable_categories.push_back(
          std::make_shared<dataset::CategoricalHistoryIndex>(
              /* n_ids = */ item.n_distinct,
              /* n_categories = */ cat.n_distinct,
              /* buffer_size = */ cat.track_last_n));
    }
    auto tracking_block = std::make_shared<dataset::CategoricalTrackingBlock>(
        item.col_num, _schema.timestamp.col_num, cat.col_num, config.lookahead,
        config.lookback, _states.item_id_map, _states.trackable_categories[i],
        item.graph, item.max_neighbors);
    blocks.push_back(tracking_block);

    size_t nonzero_multiplier = item.max_neighbors > 0 ? 2 : 1;
    addNonzeros(nonzero_multiplier * cat.track_last_n);
  }
}

}  // namespace thirdai::bolt