#include <bolt/src/sequential_classifier/PipelineBuilder.h>
#include <bolt/src/utils/AutoTuneUtils.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/blocks/Trend.h>
#include <dataset/src/encodings/categorical/StringToUidMap.h>
#include <dataset/src/encodings/count_history/CountHistoryIndex.h>
#include <memory>
#include <vector>

namespace thirdai::bolt {

PipelineBuilder::PipelineBuilder(Schema schema, const char delimiter)
    : _schema(std::move(schema)), _delimiter(delimiter) {}

std::shared_ptr<dataset::StreamingGenericDatasetLoader>
PipelineBuilder::buildPipelineForFile(std::string& filename, bool shuffle,
                                      bool overwrite_index) {
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

std::string PipelineBuilder::getHeader(dataset::DataLoader& loader) {
  auto header = loader.getHeader();
  if (!header) {
    throw std::invalid_argument(
        "[SequentialClassifier::train] The file has no header.");
  }
  return *header;
}

Blocks PipelineBuilder::buildInputBlocks(bool overwrite_index) {
  std::vector<std::shared_ptr<dataset::Block>> blocks;
  addDateFeats(blocks);
  addItemIdFeats(blocks);
  addTextAttrFeats(blocks);
  addCategoricalAttrFeats(blocks);
  addTrackableQtyFeats(blocks, overwrite_index);

  return blocks;
}

Blocks PipelineBuilder::buildLabelBlocks() {
  auto target = _schema.target;
  if (_states.target_id_map == nullptr) {
    _states.target_id_map =
        std::make_shared<dataset::StringToUidMap>(target.n_distinct);
  }

  return {std::make_shared<dataset::CategoricalBlock>(target.col_num,
                                                      _states.target_id_map)};
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

void PipelineBuilder::addDateFeats(Blocks& blocks) {
  blocks.push_back(std::make_shared<dataset::DateBlock>(_schema.timestamp.col_num));
  addNonzeros(4);
}

void PipelineBuilder::addItemIdFeats(Blocks& blocks) {
  auto item = _schema.item;
  if (_states.item_id_map == nullptr) {
    _states.item_id_map =
        std::make_shared<dataset::StringToUidMap>(item.n_distinct);
  }

  blocks.push_back(std::make_shared<dataset::CategoricalBlock>(
      item.col_num, _states.item_id_map, item.graph, item.max_neighbors));
  addNonzeros(1);
}

void PipelineBuilder::addTextAttrFeats(Blocks& blocks) {
  auto text_attrs = _schema.text_attributes;
  for (const auto& text : text_attrs) {
    blocks.push_back(
        std::make_shared<dataset::TextBlock>(text.col_num, /* dim = */ 100000));
    addNonzeros(20);
  }
}

void PipelineBuilder::addCategoricalAttrFeats(Blocks& blocks) {
  auto cat_attrs = _schema.categorical_attributes;
  for (uint32_t i = 0; i < cat_attrs.size(); i++) {
    auto cat = cat_attrs[i];
    if (i >= _states.cat_attr_maps.size()) {
      _states.cat_attr_maps.push_back(std::make_shared<dataset::StringToUidMap>(cat.n_distinct));
    }
    blocks.push_back(std::make_shared<dataset::CategoricalBlock>(
        cat.col_num, _states.cat_attr_maps[i]));
    addNonzeros(1);
  }
}

void PipelineBuilder::addTrackableQtyFeats(Blocks& blocks,
                                           bool overwrite_index) {
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
        config.horizon, config.lookback, _states.trackable_counts[i],
        item.graph, item.max_neighbors);
    blocks.push_back(trend_block);
    addNonzeros(trend_block->featureDim());
  }
}

}  // namespace thirdai::bolt