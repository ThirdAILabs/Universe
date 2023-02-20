#pragma once

#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
namespace thirdai::automl::data {

/**
 * data_types: mapping from column names (strings) to DataType objects,
 *   e.g. {"user_id_column": types.categorical()}
 *
 * graph_file_name: path to file from which we can generate static graph.
 *
 * source: column with node id's.
 * target: column name of target variable.
 *
 * n_target_classes: number of target classes.
 *
 * num_neighbours: number of neighbours we want to consider for input of
 * embedding node.
 *
 * relationship_columns: columns using which we are creating adjacency list.
 *
 * numerical_context: consider the numerical values of neighbours.
 *
 * feature_context: consider remaining features if any from the neighbours.
 *
 * k_hop: how much far you want to consider the neighbourhood.
 *
 * adj_list: adj_list of the graph if provided we bypass calculation of
 * adj_list.
 */
struct GraphConfig {
 public:
  GraphConfig(
      ColumnDataTypes data_types, std::string graph_file_name,
      std::string source, std::string target, uint32_t n_target_classes,
      uint32_t num_neighbours,
      std::optional<std::vector<std::string>> relationship_columns =
          std::nullopt,
      bool integer_target = false, bool numerical_context = false,
      bool features_context = false, uint32_t k_hop = 1, char delimeter = ',',
      std::optional<std::unordered_map<std::string, std::vector<std::string>>>
          adj_list = std::nullopt)
      : _data_types(std::move(data_types)),
        _graph_file_name(std::move(graph_file_name)),
        _source(std::move(source)),
        _target(std::move(target)),
        _num_neighbours(num_neighbours),
        _relationship_columns(std::move(relationship_columns)),
        _n_target_classes(n_target_classes),
        _integer_target(integer_target),
        _numerical_context(numerical_context),
        _features_context(features_context),
        _k_hop(k_hop),
        _delimeter(delimeter),
        _adj_list(std::move(adj_list)) {
    if (!_relationship_columns && !_adj_list) {
      throw std::invalid_argument(
          "At least one of relationship columns or adj_list has to be "
          "provided.");
    }
  }

  ColumnDataTypes _data_types;
  std::string _graph_file_name;
  std::string _source;
  std::string _target;
  uint32_t _num_neighbours;
  std::optional<std::vector<std::string>> _relationship_columns;
  uint32_t _n_target_classes;
  bool _integer_target;
  bool _numerical_context;
  bool _features_context;
  uint32_t _k_hop;
  char _delimeter;
  std::optional<std::unordered_map<std::string, std::vector<std::string>>>
      _adj_list;
};

using GraphConfigPtr = std::shared_ptr<GraphConfig>;

}  // namespace thirdai::automl::data