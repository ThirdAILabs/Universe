#pragma once

#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
namespace thirdai::automl::data {

struct GraphConfig {
 public:
  GraphConfig(
      ColumnDataTypes data_types, std::string graph_file_name,
      std::string source, std::string target, uint32_t n_target_classes,
      uint32_t max_neighbours,
      std::optional<std::vector<std::string>> relationship_columns =
          std::nullopt,
      bool numerical_context = false, bool features_context = false,
      uint32_t k_hop = 1, char delimeter = ',',
      std::optional<std::unordered_map<std::string, std::vector<std::string>>>
          adj_list = std::nullopt)
      : _data_types(std::move(data_types)),
        _graph_file_name(std::move(graph_file_name)),
        _source(std::move(source)),
        _target(std::move(target)),
        _max_neighbours(max_neighbours),
        _relationship_columns(std::move(relationship_columns)),
        _n_target_classes(n_target_classes),
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
  uint32_t _max_neighbours;
  std::optional<std::vector<std::string>> _relationship_columns;
  uint32_t _n_target_classes;
  bool _numerical_context;
  bool _features_context;
  uint32_t _k_hop;
  char _delimeter;
  std::optional<std::unordered_map<std::string, std::vector<std::string>>>
      _adj_list;
};

using GraphConfigPtr = std::shared_ptr<GraphConfig>;

}  // namespace thirdai::automl::data