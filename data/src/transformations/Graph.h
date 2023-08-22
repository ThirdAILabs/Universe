#pragma once

#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class GraphBuilder final : public Transformation {
 public:
  GraphBuilder(std::string node_id_column, std::string neighbors_column,
               std::vector<std::string> feature_columns);

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _node_id_column;
  std::string _neighbors_column;
  std::vector<std::string> _feature_columns;
};

class NeighborIds final : public Transformation {
 public:
  NeighborIds(std::string node_id_column, std::string output_neighbors_column);

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _node_id_column;
  std::string _output_neighbors_column;
};

class NeighborFeatures final : public Transformation {
 public:
  NeighborFeatures(std::string node_id_column,
                   std::string output_feature_column);

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _node_id_column;
  std::string _output_features_column;
};

}  // namespace thirdai::data