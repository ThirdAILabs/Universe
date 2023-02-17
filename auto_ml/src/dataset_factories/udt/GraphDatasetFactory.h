#pragma once

#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/dataset_factories/DatasetFactory.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/FeatureComposer.h>
#include <auto_ml/src/dataset_factories/udt/UDTDatasetFactory.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/DenseArray.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/blocks/TabularHashFeatures.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <dataset/src/utils/PreprocessedVectors.h>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
namespace thirdai::automl::data {

// TODO(Josh): Consider adding back k_hop
class GraphDatasetFactory : public DatasetLoaderFactory {
 public:
  explicit GraphDatasetFactory(data::ColumnDataTypes data_types,
                               std::string target_col,
                               uint32_t n_target_classes, char delimiter,
                               uint32_t max_neighbors,
                               bool store_node_features);

  dataset::DatasetLoaderPtr getLabeledDatasetLoader(
      std::shared_ptr<dataset::DataSource> data_source, bool training) final;

  std::vector<BoltVector> featurizeInput(const LineInput& input) final {
    (void)input;
    throw exceptions::NotImplemented(
        "Predict is not yet implemented for graph neural networks");
  }

  std::vector<BoltVector> featurizeInput(const MapInput& input) final {
    (void)input;
    throw exceptions::NotImplemented(
        "Predict is not yet implemented for graph neural networks");
  }

  std::vector<BoltBatch> featurizeInputBatch(
      const LineInputBatch& inputs) final {
    (void)inputs;
    throw exceptions::NotImplemented(
        "Predict is not yet implemented for graph neural networks");
  }

  std::vector<BoltBatch> featurizeInputBatch(
      const MapInputBatch& inputs) final {
    (void)inputs;
    throw exceptions::NotImplemented(
        "Predict is not yet implemented for graph neural networks");
  }

  uint32_t labelToNeuronId(std::variant<uint32_t, std::string> label) final {
    (void)label;
    throw exceptions::NotImplemented(
        "Explain is not yet implemented for graph neural networks");
  }

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio,
      const std::string& sample) final {
    (void)gradients_indices;
    (void)gradients_ratio;
    (void)sample;
    throw exceptions::NotImplemented(
        "Explain is not yet implemented for graph neural networks");
  };

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio, const MapInput& sample) final {
    (void)gradients_indices;
    (void)gradients_ratio;
    (void)sample;
    throw exceptions::NotImplemented(
        "Explain is not yet implemented for graph neural networks");
  }

  std::vector<uint32_t> getInputDims() final {
    return _featurizer->getDimensions();
  };

  uint32_t getLabelDim() final { return _n_target_classes; };

  bool hasTemporalTracking() const final { return false; }

 private:
  data::ColumnDataTypes _data_types;
  std::string _target_col;
  uint32_t _n_target_classes;
  char _delimiter;
  uint32_t _max_neighbors;
  bool _store_node_features;
  dataset::TabularFeaturizerPtr _graph_builder;
  dataset::TabularFeaturizerPtr _featurizer;
  GraphInfoPtr _graph_info;
};

using GraphDatasetFactoryPtr = std::shared_ptr<GraphDatasetFactory>;

}  // namespace thirdai::automl::data