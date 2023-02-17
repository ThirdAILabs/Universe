#pragma once

#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/TemporalContext.h>
#include <auto_ml/src/dataset_factories/udt/TemporalRelationshipsAutotuner.h>
#include <auto_ml/src/featurization/TabularBlockComposer.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>

namespace thirdai::automl::data {

class TabularDatasetFactory {
 public:
  TabularDatasetFactory(
      const ColumnDataTypes& input_data_types,
      const UserProvidedTemporalRelationships& provided_temporal_relationships,
      const std::vector<dataset::BlockPtr>& label_blocks,
      const TabularOptions& options, bool force_parallel);

  dataset::DatasetLoaderPtr getDatasetLoader(
      const dataset::DataSourcePtr& data_source, bool training);

  std::vector<BoltVector> featurizeInput(const MapInput& input) {
    dataset::MapSampleRef input_ref(input);
    return {_inference_featurizer->makeInputVector(input_ref)};
  }

  std::vector<BoltBatch> featurizeInputBatch(const MapInputBatch& inputs) {
    dataset::MapBatchRef inputs_ref(inputs);

    std::vector<BoltBatch> result;
    for (auto& batch : _inference_featurizer->featurize(inputs_ref)) {
      result.emplace_back(std::move(batch));
    }

    return result;
  }

  void updateTemporalTrackers(const MapInput& input) {
    dataset::MapSampleRef input_ref(input);
    _labeled_featurizer->makeInputVector(input_ref);
  }

  void updateTemporalTrackersBatch(const MapInputBatch& inputs) {
    dataset::MapBatchRef inputs_ref(inputs);
    _labeled_featurizer->featurize(inputs_ref);
  }

  void resetTemporalTrackers() { _temporal_context.reset(); }

  void updateMetadata(const std::string& col_name, const MapInput& update);

  void updateMetadataBatch(const std::string& col_name,
                           const MapInputBatch& updates);

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio, const MapInput& sample) {
    dataset::MapSampleRef sample_ref(sample);
    return bolt::getSignificanceSortedExplanations(
        gradients_indices, gradients_ratio, sample_ref, _inference_featurizer);
  }

  bool hasTemporalRelationships() const { return !_temporal_context.empty(); }

  uint32_t inputDim() const { return _labeled_featurizer->getInputDim(); }

  char delimiter() const { return _delimiter; }

  const ColumnDataTypes& inputDataTypes() const { return _input_data_types; }

 private:
  dataset::TabularFeaturizerPtr makeFeaturizer(
      const ColumnDataTypes& input_data_types,
      const TemporalRelationships& temporal_relationships,
      bool should_update_history, const TabularOptions& options,
      const std::vector<dataset::BlockPtr>& label_blocks, bool parallel);

  PreprocessedVectorsMap processAllMetadata(
      const ColumnDataTypes& input_data_types, const TabularOptions& options);

  dataset::PreprocessedVectorsPtr makeProcessedVectorsForCategoricalColumn(
      const std::string& col_name, const CategoricalDataTypePtr& categorical,
      const TabularOptions& options);

  auto getColumnMetadataConfig(const std::string& col_name) {
    return asCategorical(_input_data_types.at(col_name))->metadata_config;
  }

  void verifyColumnMetadataExists(const std::string& col_name) {
    if (!_input_data_types.count(col_name) ||
        !asCategorical(_input_data_types.at(col_name)) ||
        !asCategorical(_input_data_types.at(col_name))->metadata_config ||
        !_metadata_processors.count(col_name) ||
        !_vectors_map.count(col_name)) {
      throw std::invalid_argument("'" + col_name + "' is an invalid column.");
    }
  }

  dataset::TabularFeaturizerPtr _labeled_featurizer;
  dataset::TabularFeaturizerPtr _inference_featurizer;

  std::unordered_map<std::string, dataset::TabularFeaturizerPtr>
      _metadata_processors;

  PreprocessedVectorsMap _vectors_map;

  TemporalContext _temporal_context;

  ColumnDataTypes _input_data_types;
  char _delimiter;
};

using TabularDatasetFactoryPtr = std::shared_ptr<TabularDatasetFactory>;

}  // namespace thirdai::automl::data