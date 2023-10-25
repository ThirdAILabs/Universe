#pragma once

#include "BlockInterface.h"
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/utils/GraphInfo.h>
#include <exceptions/src/Exceptions.h>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace thirdai::dataset {

/** Sums and normalizes the numerical features of a node's neighbors */
class NormalizedNeighborVectorsBlock final : public Block {
 public:
  explicit NormalizedNeighborVectorsBlock(ColumnIdentifier node_id_col,
                                          automl::GraphInfoPtr graph_ptr)
      : _node_id_col(std::move(node_id_col)),
        _graph_ptr(std::move(graph_ptr)) {}

  uint32_t featureDim() const final { return _graph_ptr->featureDim(); }

  bool isDense() const final { return true; }

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input) final;

  static auto make(ColumnIdentifier col,
                   const automl::GraphInfoPtr& graph_ptr) {
    return std::make_shared<NormalizedNeighborVectorsBlock>(std::move(col),
                                                            graph_ptr);
  }

 protected:
  void buildSegment(ColumnarInputSample& input,
                    SegmentedFeatureVector& vec) final;

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final {
    return {&_node_id_col};
  }

 private:
  ColumnIdentifier _node_id_col;
  automl::GraphInfoPtr _graph_ptr;

  NormalizedNeighborVectorsBlock() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);
};

/** Returns a sparse vector corresponding to a node's neighbors. */
class NeighborTokensBlock final : public Block {
 public:
  explicit NeighborTokensBlock(ColumnIdentifier node_id_col,
                               automl::GraphInfoPtr graph_ptr)
      : _node_id_col(std::move(node_id_col)),
        _graph_ptr(std::move(graph_ptr)) {}

  // This is a bit of a hack/leaky abstraction, because if we ever try to append
  // this block with other blocks it will overflow. However, this block should
  // only ever be used on it's own (and this is a pattern we already use in the
  // TabularHashFeatures block)
  uint32_t featureDim() const final {
    return std::numeric_limits<uint32_t>::max();
  }

  bool isDense() const final { return false; }

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input) final;

  static auto make(ColumnIdentifier col,
                   const automl::GraphInfoPtr& graph_ptr) {
    return std::make_shared<NeighborTokensBlock>(std::move(col), graph_ptr);
  }

 protected:
  void buildSegment(ColumnarInputSample& input,
                    SegmentedFeatureVector& vec) final;

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final {
    return {&_node_id_col};
  }

 private:
  ColumnIdentifier _node_id_col;
  automl::GraphInfoPtr _graph_ptr;

  NeighborTokensBlock() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);
};

/** Populates the passed in GraphInfoPtr with adjacency and node feature data */
class GraphBuilderBlock final : public Block {
 public:
  explicit GraphBuilderBlock(ColumnIdentifier neighbor_col,
                             ColumnIdentifier node_id_col,
                             std::vector<ColumnIdentifier> feature_cols,
                             automl::GraphInfoPtr graph_ptr)
      : _node_id_col(std::move(node_id_col)),
        _neighbor_col(std::move(neighbor_col)),
        _feature_cols(std::move(feature_cols)),
        _graph_ptr(std::move(graph_ptr)) {}

  // This is 0 because we are not adding anything to the vector, only adding to
  // the passed in graph info pointer
  uint32_t featureDim() const final { return 0; }

  bool isDense() const final {
    // featureDim of 0 is "dense"
    return true;
  }

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input) final;

  static auto make(ColumnIdentifier neighbor_col, ColumnIdentifier node_id_col,
                   std::vector<ColumnIdentifier> feature_cols,
                   const automl::GraphInfoPtr& graph_ptr) {
    return std::make_shared<GraphBuilderBlock>(
        std::move(neighbor_col), std::move(node_id_col),
        std::move(feature_cols), graph_ptr);
  }

 protected:
  void buildSegment(ColumnarInputSample& input,
                    SegmentedFeatureVector& vec) final;

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final {
    std::vector<ColumnIdentifier*> column_identifiers;
    column_identifiers.reserve(_feature_cols.size());
    for (auto& _feature_col : _feature_cols) {
      column_identifiers.push_back(&_feature_col);
    }
    column_identifiers.push_back(&_node_id_col);
    column_identifiers.push_back(&_neighbor_col);
    return column_identifiers;
  }

 private:
  ColumnIdentifier _node_id_col, _neighbor_col;
  std::vector<ColumnIdentifier> _feature_cols;
  automl::GraphInfoPtr _graph_ptr;

  GraphBuilderBlock() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::dataset