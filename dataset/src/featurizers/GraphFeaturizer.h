#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <sys/types.h>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>

namespace thirdai::dataset {

/**
 * This featurizer creates two input vectors and a label vector for each row in
 * the data. It uses input blocks to create an input for Input node, and it will
 * use the neighbours to create token input for Token input node and label block
 * for label vector.
 *
 * If the number of neighbours for a node is less than the max neighbours(number
 * of neighbours we want to consider for token input) then we add remaining with
 * null node.
 */

class GraphFeaturizer final : public Featurizer {
 public:
  GraphFeaturizer(std::vector<std::shared_ptr<Block>> input_blocks,
                  std::vector<std::shared_ptr<Block>> label_blocks,
                  ColumnIdentifier source_col, uint32_t max_neighbors,
                  char delimiter = ',',
                  std::optional<uint32_t> hash_range = std::nullopt)
      : _input_blocks(std::move(input_blocks)),
        _label_blocks(std::move(label_blocks)),
        _source_col(std::move(source_col)),
        _max_neighbors(max_neighbors),
        _delimiter(delimiter),
        _hash_range(hash_range) {
    _node_id_to_num_map.insert({"null_node", 0});
  }

  static std::shared_ptr<GraphFeaturizer> make(
      std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks,
      ColumnIdentifier source_col, uint32_t max_neighbors,
      char delimiter = ',', std::optional<uint32_t> hash_range = std::nullopt) {
    return std::make_shared<GraphFeaturizer>(
        std::move(input_blocks), std::move(label_blocks), std::move(source_col),
        max_neighbors, delimiter, hash_range);
  }

  std::vector<std::vector<BoltVector>> featurize(
      const LineInputBatch& input_batch) final;

  std::vector<std::vector<BoltVector>> featurize(
      ColumnarInputBatch& input_batch);

  uint32_t getInputDim() const {
    return _hash_range.value_or(_input_blocks.featureDim());
  }

  uint32_t getLabelDim() const { return _label_blocks.featureDim(); }

  bool expectsHeader() const final { return true; }

  void processHeader(const std::string& header) final;

  size_t getNumDatasets() final { return 3; }

  void updateNeighbours(const automl::Neighbours& neighbours);

  void updateNodeIdMap(const ColumnNumberMap& node_id_map);

 private:
  std::exception_ptr featurizeSampleInBatch(
      uint32_t index_in_batch, ColumnarInputBatch& input_batch,
      std::vector<BoltVector>& batch_inputs,
      std::vector<BoltVector>& batch_token_inputs,
      std::vector<BoltVector>& batch_labels);

  BoltVector buildNeighbourVector(ColumnarInputSample& sample);

  automl::Neighbours _neighbours;
  std::unordered_map<std::string, uint32_t> _node_id_to_num_map;
  BlockList _input_blocks;
  BlockList _label_blocks;
  ColumnIdentifier _source_col;
  uint32_t _expected_num_cols;
  uint32_t _max_neighbors;
  char _delimiter;
  std::optional<uint32_t> _hash_range;
};

using GraphFeaturizerPtr = std::shared_ptr<GraphFeaturizer>;

}  // namespace thirdai::dataset