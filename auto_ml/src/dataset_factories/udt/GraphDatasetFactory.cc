#include "GraphDatasetFactory.h"
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/dataset_factories/udt/DatasetFactoryUtils.h>
#include <dataset/src/blocks/TabularHashFeatures.h>
#include <stdexcept>

namespace thirdai::automl::data {

GraphDatasetFactory::GraphDatasetFactory(data::ColumnDataTypes data_types,
                                         std::string target_col,
                                         uint32_t n_target_classes,
                                         char delimiter, uint32_t max_neighbors,
                                         uint32_t k_hop,
                                         bool store_node_features)
    : _data_types(std::move(data_types)),
      _target_col(std::move(target_col)),
      _n_target_classes(n_target_classes),
      _delimiter(delimiter),
      _max_neighbors(max_neighbors),
      _k_hop(k_hop),
      _store_node_features(store_node_features) {
  if (_data_types.count("neighbors") == 0) {
    throw std::invalid_argument(
        "There must be a neighbors column to use a graph neural network.");
  }

  if (_k_hop > 3 || _k_hop == 0) {
    throw std::invalid_argument("K hop must be between 1 and 3 inclusive.");
  }
}

}  // namespace thirdai::automl::data