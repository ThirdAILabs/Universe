#pragma once

#include "BlockInterface.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>

namespace thirdai::dataset {

/* Represents a list of blocks that can featurize input samples into a vector */
struct BlockList {
  explicit BlockList(std::vector<BlockPtr>&& blocks,
                     /*
                     If hash_range has a value, then features from different
                     blocks will be aggregated by hashing them to the same range
                     but with different hash salts. Otherwise, the features will
                     be treated as sparse vectors, which are then concatenated.
                   */
                     std::optional<uint32_t> hash_range = std::nullopt);

  BlockList() {}

  auto operator[](uint32_t index) { return _blocks[index]; }

  const auto& blocks() const { return _blocks; }

  const auto& hashRange() const { return _hash_range; }

  /**
   * Dispatches the method each Block. See method definition in the
   * Block class for details.
   */
  void updateColumnNumbers(const ColumnNumberMap& column_number_map);

  /**
   * Dispatches the method each Block. See method definition in the
   * Block class for details.
   */
  void prepareForBatch(ColumnarInputBatch& incoming_batch);

  uint32_t featureDim() const { return _feature_dim; }

  uint32_t expectedNumColumns() const { return _expected_num_columns; }

  std::shared_ptr<SegmentedFeatureVector> buildVector(
      ColumnarInputSample& sample, bool store_segment_feature_map = false);

 private:
  std::shared_ptr<SegmentedFeatureVector> makeSegmentedFeatureVector(
      bool store_segment_feature_map);

  static bool computeAreDense(const std::vector<BlockPtr>& blocks);

  static bool allBlocksHaveColumnNumbers(const std::vector<BlockPtr>& blocks);

  static uint32_t computeExpectedNumColumns(
      const std::vector<BlockPtr>& blocks);

  static uint32_t computeFeatureDim(const std::vector<BlockPtr>& blocks);

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_blocks, _are_dense, _expected_num_columns, _feature_dim,
            _hash_range);
  }

  std::vector<BlockPtr> _blocks;
  bool _are_dense;
  uint32_t _feature_dim;
  uint32_t _expected_num_columns;
  std::optional<uint32_t> _hash_range;
};

using BlockListPtr = std::shared_ptr<BlockList>;

}  // namespace thirdai::dataset