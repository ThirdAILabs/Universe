#pragma once

#include <data/src/transformations/Transformation.h>
#include <proto/graph.pb.h>
#include <memory>
#include <utility>

namespace thirdai::data {

class GraphBuilder final : public Transformation {
 public:
  GraphBuilder(std::string node_id_column, std::string neighbors_column,
               std::vector<std::string> feature_columns);

  static std::shared_ptr<GraphBuilder> make(
      std::string node_id_column, std::string neighbors_column,
      std::vector<std::string> feature_columns) {
    return std::make_shared<GraphBuilder>(std::move(node_id_column),
                                          std::move(neighbors_column),
                                          std::move(feature_columns));
  }

  explicit GraphBuilder(const proto::data::GraphBuilder& graph_builder);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  proto::data::Transformation* toProto() const final;

 private:
  std::string _node_id_column;
  std::string _neighbors_column;
  std::vector<std::string> _feature_columns;
};

class NeighborIds final : public Transformation {
 public:
  NeighborIds(std::string node_id_column, std::string output_neighbors_column);

  explicit NeighborIds(const proto::data::NeighborIds& nbr_ids);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  proto::data::Transformation* toProto() const final;

 private:
  std::string _node_id_column;
  std::string _output_neighbors_column;
};

class NeighborFeatures final : public Transformation {
 public:
  NeighborFeatures(std::string node_id_column,
                   std::string output_feature_column);

  explicit NeighborFeatures(const proto::data::NeighborFeatures& nbr_features);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  proto::data::Transformation* toProto() const final;

 private:
  std::string _node_id_column;
  std::string _output_features_column;
};

}  // namespace thirdai::data