#include "GraphDatasetFactory.h"
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
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
  if (_data_types.count("neighbors") == 0 ||
      !asNeighbors(_data_types.at("neighbors"))) {
    throw std::invalid_argument(
        "There must be a neighbors column with data type "
        "bolt.types.neighbors() to use a graph neural network.");
  }

  if (_k_hop > 3 || _k_hop == 0) {
    throw std::invalid_argument("K hop must be between 1 and 3 inclusive.");
  }
}

dataset::DatasetLoaderPtr GraphDatasetFactory::getLabeledDatasetLoader(
    std::shared_ptr<dataset::DataSource> data_source, bool training) {
  auto column_number_map = DatasetFactoryUtils::makeColumnNumberMapFromHeader(
      *data_source, _delimiter);
  // std::vector<std::string> column_number_to_name =
  // column_number_map.getColumnNumToColNameMap();

  // // The featurizer will treat the next line as a header
  // // Restart so featurizer does not skip a sample.
  // data_source->restart();

  // _featurizer->updateColumnNumbers(column_number_map);

  return std::make_unique<dataset::DatasetLoader>(data_source, _featurizer,
                                                  /* shuffle= */ training);
}

}  // namespace thirdai::automl::data