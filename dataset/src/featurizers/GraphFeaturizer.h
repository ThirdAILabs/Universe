#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
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
 *
 * Here node_id_map expects the node number starts from 1 because we insert a
 * null node with number 0.
 */

class GraphFeaturizer final : public Featurizer {
 public:
  GraphFeaturizer(std::vector<std::shared_ptr<Block>> input_blocks,
                  std::vector<std::shared_ptr<Block>> label_blocks,
                  ColumnIdentifier source_col, uint32_t num_neighbours,
                  Neighbours neighbours,
                  std::unordered_map<std::string, uint32_t> node_id_map,
                  char delimiter = ',',
                  std::optional<uint32_t> hash_range = std::nullopt)
      : _tabular_featurizer(std::move(input_blocks), std::move(label_blocks),
                            /*has_header=*/false, delimiter, /*parallel=*/true,
                            hash_range),
        _neighbours(std::move(neighbours)),
        _node_id_to_num_map(std::move(node_id_map)),
        _source_col(std::move(source_col)),
        _num_neighbours(num_neighbours),
        _delimiter(delimiter) {
    _node_id_to_num_map.insert({"null_node", 0});
  }

  static std::shared_ptr<GraphFeaturizer> make(
      std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks,
      ColumnIdentifier source_col, uint32_t num_neighbours,
      Neighbours neighbours,
      std::unordered_map<std::string, uint32_t> node_id_map,
      char delimiter = ',', std::optional<uint32_t> hash_range = std::nullopt) {
    return std::make_shared<GraphFeaturizer>(
        std::move(input_blocks), std::move(label_blocks), std::move(source_col),
        num_neighbours, std::move(neighbours), std::move(node_id_map),
        delimiter, hash_range);
  }

  std::vector<std::vector<BoltVector>> featurize(
      const LineInputBatch& input_batch) final;

  std::vector<std::vector<BoltVector>> featurize(
      ColumnarInputBatch& input_batch);

  uint32_t getInputDim() const { return _tabular_featurizer.getInputDim(); }

  uint32_t getLabelDim() const { return _tabular_featurizer.getLabelDim(); }

  bool expectsHeader() const final { return true; }

  void processHeader(const std::string& header) final;

  size_t getNumDatasets() final { return 3; }

 private:
  std::exception_ptr featurizeSampleInBatch(
      uint32_t index_in_batch, ColumnarInputBatch& input_batch,
      std::vector<BoltVector>& batch_inputs,
      std::vector<BoltVector>& batch_token_inputs,
      std::vector<BoltVector>& batch_labels);

  BoltVector buildNeighbourVector(ColumnarInputSample& sample);

  TabularFeaturizer _tabular_featurizer;
  Neighbours _neighbours;
  std::unordered_map<std::string, uint32_t> _node_id_to_num_map;
  ColumnIdentifier _source_col;
  uint32_t _num_neighbours;
  char _delimiter;
  uint32_t _expected_num_cols;
};

using GraphFeaturizerPtr = std::shared_ptr<GraphFeaturizer>;

}  // namespace thirdai::dataset