#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <_types/_uint32_t.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <optional>
#include <unordered_map>
#include <utility>

namespace thirdai::dataset {

class GraphFeaturizer final : public Featurizer {
 public:
  GraphFeaturizer(uint32_t k_hop,
                  std::vector<ColumnIdentifier> numerical_columns,
                  std::shared_ptr<Block> label_block,
                  std::vector<ColumnIdentifier>
                      relationship_columns,
                  char delimiter = ',')
      : _k_hop(k_hop),
        _numerical_columns(std::move(numerical_columns)),
        _relationship_columns(std::move(relationship_columns)),
        _label_block(std::move(label_block)),
        _delimiter(delimiter) {}

  std::vector<std::vector<BoltVector>> featurize(
      const std::vector<std::string>& rows) final;

  bool expectsHeader() const final { return true; }

  void processHeader(const std::string& header) final;

  size_t getNumDatasets() final { return 3; }

  void updateAdjacencyList(
      const std::unordered_map<std::string, std::vector<std::string>>&
          adj_list);

 private:
  std::tuple<BoltVector, BoltVector, BoltVector> processRow(
      const std::string& row) const;

  void addNodeInfo(const std::string& node_id,
                   const std::vector<std::string>& neighbours);

  uint32_t _k_hop;
  std::vector<ColumnIdentifier> _numerical_columns;
  std::unordered_map<std::string, std::vector<std::string>> _adjacency_list;
  std::vector<ColumnIdentifier> _relationship_columns;
  std::shared_ptr<Block> _label_block;
  char _delimiter;
};

}  // namespace thirdai::dataset