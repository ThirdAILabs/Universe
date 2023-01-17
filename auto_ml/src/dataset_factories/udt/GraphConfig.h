#pragma once

#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <optional>
#include <utility>
namespace thirdai::automl::data {

struct GraphConfig {
 public:
  GraphConfig(ColumnDataTypes data_types, std::string graph_file_name,
              std::string source, std::string target,
              std::vector<std::string> relationship_columns,
              uint32_t batch_size, uint32_t n_target_classes,
              bool neighbourhood_context = false, bool label_context = false,
              uint32_t kth_neighbourhood = 0, char delimeter = ',')
      : _data_types(std::move(data_types)),
        _graph_file_name(std::move(graph_file_name)),
        _source(std::move(source)),
        _target(std::move(target)),
        _relationship_columns(std::move(relationship_columns)),
        _batch_size(batch_size),
        _n_target_classes(n_target_classes),
        _neighbourhood_context(neighbourhood_context),
        _label_context(label_context),
        _kth_neighbourhood(kth_neighbourhood),
        _delimeter(delimeter) {}

  ColumnDataTypes _data_types;
  std::string _graph_file_name;
  std::string _source;
  std::string _target;
  std::vector<std::string> _relationship_columns;
  uint32_t _batch_size;
  uint32_t _n_target_classes;
  bool _neighbourhood_context;
  bool _label_context;
  uint32_t _kth_neighbourhood;
  char _delimeter;
};

using GraphConfigPtr = std::shared_ptr<GraphConfig>;

}  // namespace thirdai::automl::data