#pragma once
#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/dataset_factories/DatasetFactory.h>
#include <auto_ml/src/dataset_factories/udt/CategoricalMetadata.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Sequence.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <cstdint>
#include <memory>

namespace thirdai::automl::data {

class RNNDatasetFactory;

using RNNDatasetFactoryPtr = std::shared_ptr<RNNDatasetFactory>;

class RNNDatasetFactory final : public DatasetLoaderFactory {
 public:
  static std::shared_ptr<RNNDatasetFactory> make(
      ColumnDataTypes data_types, std::string target_column,
      uint32_t n_target_classes, char delimiter,
      uint32_t text_pairgram_word_limit, bool contextual_columns,
      uint32_t hash_range);

  dataset::DatasetLoaderPtr getLabeledDatasetLoader(
      dataset::DataSourcePtr data_source, bool training) final;

  std::vector<BoltVector> featurizeInput(const MapInput& input) final {
    dataset::MapSampleRef input_ref(input);
    return {_unlabeled_featurizer->makeInputVector(input_ref)};
  }

  std::vector<BoltBatch> featurizeInputBatch(
      const MapInputBatch& inputs) final {
    dataset::MapBatchRef inputs_ref(inputs);

    auto batches = _unlabeled_featurizer->featurize(inputs_ref);

    // We cannot use the initializer list because the copy constructor is
    // deleted for BoltBatch.
    std::vector<BoltBatch> batch_list;
    batch_list.emplace_back(std::move(batches.at(0)));
    return batch_list;
  }

  uint32_t labelToNeuronId(std::variant<uint32_t, std::string> label) final;

  std::string className(uint32_t neuron_id) const final;

  void incorporateNewPrediction(MapInput& sample,
                                const std::string& new_prediction) const;

  std::string stitchTargetSequence(
      const std::vector<std::string>& predictions) const;

  void updateMetadata(const std::string& col_name, const MapInput& update) {
    _categorical_metadata.updateMetadata(col_name, update);
  }

  void updateMetadataBatch(const std::string& col_name,
                           const MapInputBatch& updates) {
    _categorical_metadata.updateMetadataBatch(col_name, updates);
  }

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio, const MapInput& sample) final {
    (void)gradients_indices;
    (void)gradients_ratio;
    (void)sample;
    throw std::invalid_argument(
        "Recursive model currently does not support explanation.");
  }

  std::vector<uint32_t> getInputDims() final {
    return {_labeled_featurizer->getInputDim()};
  }

  uint32_t getLabelDim() final { return _labeled_featurizer->getLabelDim(); }

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static RNNDatasetFactoryPtr load(const std::string& filename);

  static RNNDatasetFactoryPtr load_stream(std::istream& input_stream);

  bool hasTemporalTracking() const final { return false; }

 private:
  // This private constructor is only called by make()
  explicit RNNDatasetFactory(ColumnDataTypes augmented_data_types,
                             std::string intermediate_column,
                             std::string current_step_target_column,
                             std::string step_column, char delimiter,
                             char target_sequence_delimiter,
                             dataset::SequenceTargetBlockPtr label_block,
                             CategoricalMetadata categorical_metadata,
                             dataset::TabularFeaturizerPtr unlabeled_featurizer,
                             dataset::TabularFeaturizerPtr labeled_featurizer);

  ColumnDataTypes _data_types;
  std::string _intermediate_column;
  std::string _current_step_target_column;
  std::string _step_column;
  char _delimiter;
  char _target_delimiter;

  dataset::SequenceTargetBlockPtr _label_block;
  CategoricalMetadata _categorical_metadata;

  dataset::TabularFeaturizerPtr _unlabeled_featurizer;
  dataset::TabularFeaturizerPtr _labeled_featurizer;

  // Private constructor for cereal.
  RNNDatasetFactory() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactory>(this), _data_types,
            _intermediate_column, _current_step_target_column, _step_column,
            _delimiter, _target_delimiter, _label_block, _categorical_metadata,
            _unlabeled_featurizer, _labeled_featurizer);
  }
};

}  // namespace thirdai::automl::data