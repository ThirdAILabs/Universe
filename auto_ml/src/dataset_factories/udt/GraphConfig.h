#pragma once

#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <optional>
#include <unordered_map>
#include <utility>
namespace thirdai::automl::data {

struct GraphConfig {
 public:
  GraphConfig(ColumnDataTypes data_types, std::string graph_file_name,
              std::string source, std::string target, uint32_t n_target_classes,
              std::optional<std::vector<std::string>> relationship_columns =
                  std::nullopt,
              bool neighbourhood_context = false, bool label_context = false,
              uint32_t kth_neighbourhood = 0, char delimeter = ',',
              std::optional<std::unordered_map<uint32_t, std::vector<uint32_t>>>
                  adj_list = std::nullopt)
      : _data_types(std::move(data_types)),
        _graph_file_name(std::move(graph_file_name)),
        _source(std::move(source)),
        _target(std::move(target)),
        _relationship_columns(std::move(relationship_columns)),
        _n_target_classes(n_target_classes),
        _neighbourhood_context(neighbourhood_context),
        _label_context(label_context),
        _kth_neighbourhood(kth_neighbourhood),
        _delimeter(delimeter),
        _adj_list(std::move(adj_list)) {}

  ColumnDataTypes _data_types;
  std::string _graph_file_name;
  std::string _source;
  std::string _target;
  std::optional<std::vector<std::string>> _relationship_columns;
  uint32_t _n_target_classes;
  bool _neighbourhood_context;
  bool _label_context;
  uint32_t _kth_neighbourhood;
  char _delimeter;
  std::optional<std::unordered_map<uint32_t, std::vector<uint32_t>>> _adj_list;
};

using GraphConfigPtr = std::shared_ptr<GraphConfig>;

}  // namespace thirdai::automl::data