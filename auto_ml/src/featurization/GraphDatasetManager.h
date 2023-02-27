#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/udt/utils/GraphInfo.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>

namespace thirdai::automl::data {

class GraphDatasetManager {
 public:
  GraphDatasetManager(data::ColumnDataTypes data_types, std::string target_col,
                      uint32_t n_target_classes, char delimiter,
                      bool use_pairgrams);

  // TODO(Josh): Have user call index() then getDatasetLoader()

  dataset::DatasetLoaderPtr indexAndGetDatasetLoader(
      std::shared_ptr<dataset::DataSource> data_source);

  void index(const std::shared_ptr<dataset::DataSource>& data_source);

  std::vector<uint32_t> getInputDims() const {
    return _featurizer->getDimensions();
  };

  uint32_t getLabelDim() const { return _n_target_classes; };

 private:
  data::ColumnDataTypes _data_types;
  std::string _target_col;
  uint32_t _n_target_classes;
  char _delimiter;
  dataset::TabularFeaturizerPtr _graph_builder, _featurizer;
  GraphInfoPtr _graph_info;
};

using GraphDatasetManagerPtr = std::shared_ptr<GraphDatasetManager>;

}  // namespace thirdai::automl::data