#pragma once

#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class GraphBuilder final : public Transformation {
 public:
  GraphBuilder(std::string node_id_column, std::string neighbors_column,
               std::vector<std::string> feature_columns);

  explicit GraphBuilder(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "graph_builder"; }

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

  explicit NeighborIds(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "nbr_ids"; }

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

  explicit NeighborFeatures(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "nbr_features"; }

 private:
  std::string _node_id_column;
  std::string _output_features_column;

  NeighborFeatures() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data