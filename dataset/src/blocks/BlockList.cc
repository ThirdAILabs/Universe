#include "BlockList.h"

namespace thirdai::dataset {

BlockList::BlockList(std::vector<BlockPtr>&& blocks,
                     /*
                     If hash_range has a value, then features from different
                     blocks will be aggregated by hashing them to the same range
                     but with different hash salts. Otherwise, the features will
                     be treated as sparse vectors, which are then concatenated.
                   */
                     std::optional<uint32_t> hash_range)
    : _blocks(std::move(blocks)),
      _are_dense(computeAreDense(_blocks)),
      _feature_dim(hash_range.value_or(computeFeatureDim(_blocks))),
      _expected_num_columns(allBlocksHaveColumnNumbers(_blocks)
                                ? computeExpectedNumColumns(_blocks)
                                : 0),
      _hash_range(hash_range) {}

void BlockList::updateColumnNumbers(const ColumnNumberMap& column_number_map) {
  for (const auto& block : _blocks) {
    block->updateColumnNumbers(column_number_map);
  }
  _expected_num_columns = computeExpectedNumColumns(_blocks);
}
void BlockList::prepareForBatch(ColumnarInputBatch& incoming_batch) {
  for (const auto& block : _blocks) {
    block->prepareForBatch(incoming_batch);
  }
}

std::shared_ptr<SegmentedFeatureVector> BlockList::buildVector(
    ColumnarInputSample& sample, bool store_segment_feature_map) {
  auto segmented_vector = makeSegmentedFeatureVector(store_segment_feature_map);
  addVectorSegments(sample, *segmented_vector);
  return segmented_vector;
}

std::shared_ptr<SegmentedFeatureVector> BlockList::makeSegmentedFeatureVector(
    bool store_segment_feature_map) {
  if (_hash_range) {
    return std::make_shared<HashedSegmentedFeatureVector>(
        *_hash_range, store_segment_feature_map);
  }
  // Dense vector if all blocks produce dense features, sparse vector
  // otherwise.
  if (_are_dense) {
    return std::make_shared<SegmentedDenseFeatureVector>(
        store_segment_feature_map);
  }
  return std::make_shared<SegmentedSparseFeatureVector>(
      store_segment_feature_map);
}

bool BlockList::allBlocksHaveColumnNumbers(
    const std::vector<BlockPtr>& blocks) {
  if (blocks.empty()) {
    return false;
  }

  auto first_block_has_column_numbers = blocks.front()->hasColumnNumbers();
  for (const auto& block : blocks) {
    if (block->hasColumnNumbers() != first_block_has_column_numbers) {
      throw std::invalid_argument(
          "Blocks must be either all initialized with a column name or all "
          "initialized with a column number.");
    }
  }

  return first_block_has_column_numbers;
}

uint32_t BlockList::computeExpectedNumColumns(
    const std::vector<BlockPtr>& blocks) {
  uint32_t max_expected_columns = 0;
  for (const auto& block : blocks) {
    max_expected_columns =
        std::max(max_expected_columns, block->computeExpectedNumColumns());
  }
  return max_expected_columns;
}

uint32_t BlockList::computeFeatureDim(const std::vector<BlockPtr>& blocks) {
  uint32_t dim = 0;
  for (const auto& block : blocks) {
    dim += block->featureDim();
  }
  return dim;
}

void BlockList::addVectorSegments(ColumnarInputSample& sample,
                                  SegmentedFeatureVector& segmented_vector) {
  for (auto& block : _blocks) {
    block->addVectorSegment(sample, segmented_vector);
  }
}

bool BlockList::computeAreDense(const std::vector<BlockPtr>& blocks) {
  auto are_dense = std::all_of(
      blocks.begin(), blocks.end(),
      [](const std::shared_ptr<Block>& block) { return block->isDense(); });
  return are_dense;
}

}  // namespace thirdai::dataset