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

  GraphBuilder() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

class NeighborIds final : public Transformation {
 public:
  NeighborIds(std::string node_id_column, std::string output_neighbors_column);

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _node_id_column;
  std::string _output_neighbors_column;

  NeighborIds() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

class NeighborFeatures final : public Transformation {
 public:
  NeighborFeatures(std::string node_id_column,
                   std::string output_feature_column);

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _node_id_column;
  std::string _output_features_column;

  NeighborFeatures() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data