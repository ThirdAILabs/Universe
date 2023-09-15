#pragma once

#include <cereal/access.hpp>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularBlockComposer.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <auto_ml/src/featurization/TemporalContext.h>
#include <auto_ml/src/featurization/TemporalRelationshipsAutotuner.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/BlockList.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <optional>

namespace thirdai::automl::data {

class TabularDatasetFactory {
 public:
  TabularDatasetFactory(
      ColumnDataTypes input_data_types,
      const UserProvidedTemporalRelationships& provided_temporal_relationships,
      const std::vector<dataset::BlockList>& label_blocks,
      std::set<std::string> label_col_names, const TabularOptions& options,
      bool force_parallel);

  static auto make(
      ColumnDataTypes input_data_types,
      const UserProvidedTemporalRelationships& provided_temporal_relationships,
      const std::vector<dataset::BlockList>& label_blocks,
      std::set<std::string> label_col_names, const TabularOptions& options,
      bool force_parallel) {
    return std::make_shared<TabularDatasetFactory>(
        std::move(input_data_types), provided_temporal_relationships,
        label_blocks, std::move(label_col_names), options, force_parallel);
  }

  dataset::DatasetLoaderPtr getLabeledDatasetLoader(
      const dataset::DataSourcePtr& data_source, bool shuffle,
      std::optional<dataset::DatasetShuffleConfig> shuffle_config =
          std::nullopt);

  dataset::DatasetLoaderPtr getUnLabeledDatasetLoader(
      const dataset::DataSourcePtr& data_source);

  bolt::TensorList featurizeInput(const MapInput& input) {
    for (const auto& [column_name, _] : input) {
      if (!_data_types.count(column_name)) {
        throw std::invalid_argument("Input column name '" + column_name +
                                    "' not found in data_types.");
      }
    }
    dataset::MapSampleRef input_ref(input);
    return bolt::convertVectors(_inference_featurizer->featurize(input_ref),
                                _inference_featurizer->getDimensions());
  }

  bolt::TensorList featurizeInputBatch(const MapInputBatch& inputs);

  std::pair<bolt::TensorList, bolt::TensorList> featurizeTrainingBatch(
      const MapInputBatch& batch);

  void updateTemporalTrackers(const MapInput& input) {
    dataset::MapSampleRef input_ref(input);
    _labeled_featurizer->featurize(input_ref);
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

  uint32_t inputDim() const {
    /*
      TabularDatasetFactory is used by UDTClassifier and UDTRegression,
      which both expect a single input vector. Since we configured the
      featurizer with input blocks in the first position and label blocks in
      the second position, the dimension of the input vectors is the 0-th entry
      of the return value of _labeld_featurizer->getDimensions()
    */
    return _labeled_featurizer->getDimensions().at(0);
  }

  char delimiter() const { return _options.delimiter; }

  ColumnDataTypes inputDataTypes() const {
    ColumnDataTypes input_data_types;
    for (const auto& [col_name, data_type] : _data_types) {
      if (!_label_col_names.count(col_name)) {
        input_data_types[col_name] = data_type;
      }
    }
    return input_data_types;
  }

  ColumnDataTypes dataTypes() const { return _data_types; }

  void verifyCanDistribute() {
    if (!_temporal_context.empty()) {
      throw std::invalid_argument(
          "UDT with temporal relationships cannot be trained in a distributed "
          "setting.");
    }
  }

  TabularOptions tabularOptions() { return _options; }

  void save_stream(std::ostream& output_stream) const;

  dataset::DatasetLoaderPtr makeDataLoaderCustomFeaturizer(
      const dataset::DataSourcePtr& data_source, bool shuffle,
      const dataset::FeaturizerPtr& featurizer,
      std::optional<dataset::DatasetShuffleConfig> shuffle_config =
          std::nullopt) const {
    if (!shuffle_config.has_value()) {
      shuffle_config = dataset::DatasetShuffleConfig();
    }
    auto csv_data_source =
        dataset::CsvDataSource::make(data_source, delimiter());
    return std::make_unique<dataset::DatasetLoader>(
        csv_data_source, featurizer, shuffle, shuffle_config.value());
  }

  static std::shared_ptr<TabularDatasetFactory> load_stream(
      std::istream& input_stream);

 private:
  dataset::TabularFeaturizerPtr makeFeaturizer(
      const TemporalRelationships& temporal_relationships,
      bool should_update_history, const TabularOptions& options,
      const std::vector<dataset::BlockList>& label_blocks, bool parallel);

  PreprocessedVectorsMap processAllMetadata(
      const ColumnDataTypes& input_data_types, const TabularOptions& options);

  dataset::PreprocessedVectorsPtr makeProcessedVectorsForCategoricalColumn(
      const std::string& col_name, const CategoricalDataTypePtr& categorical,
      const TabularOptions& options);

  auto getColumnMetadataConfig(const std::string& col_name) {
    return asCategorical(_data_types.at(col_name))->metadata_config;
  }

  static dataset::PreprocessedVectorsPtr preprocessedVectorsFromDataset(
      dataset::DatasetLoader& dataset_loader,
      dataset::ThreadSafeVocabulary& key_vocab);

  void verifyColumnMetadataExists(const std::string& col_name) {
    if (!_data_types.count(col_name) ||
        !asCategorical(_data_types.at(col_name)) ||
        !asCategorical(_data_types.at(col_name))->metadata_config ||
        !_metadata_processors.count(col_name) ||
        !_vectors_map.count(col_name)) {
      throw std::invalid_argument("'" + col_name + "' is an invalid column.");
    }
  }

  TabularDatasetFactory() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  dataset::TabularFeaturizerPtr _labeled_featurizer;
  dataset::TabularFeaturizerPtr _inference_featurizer;

  std::unordered_map<std::string, dataset::TabularFeaturizerPtr>
      _metadata_processors;

  PreprocessedVectorsMap _vectors_map;

  TemporalContext _temporal_context;

  ColumnDataTypes _data_types;
  std::set<std::string> _label_col_names;
  uint32_t _num_label_blocks;

  TabularOptions _options;
};

using TabularDatasetFactoryPtr = std::shared_ptr<TabularDatasetFactory>;

}  // namespace thirdai::automl::data