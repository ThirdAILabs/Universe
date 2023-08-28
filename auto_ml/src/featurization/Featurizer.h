#pragma once

#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/rca/ExplanationMap.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <proto/featurizers.pb.h>
#include <stdexcept>

namespace thirdai::automl {

class TextDatasetConfig {
 public:
  TextDatasetConfig(std::string text_column, std::string label_column,
                    std::optional<char> label_delimiter)
      : _text_column(std::move(text_column)),
        _label_column(std::move(label_column)),
        _label_delimiter(label_delimiter) {}

  explicit TextDatasetConfig(const proto::udt::TextDatasetConfig& text_dataset);

  const auto& textColumn() const { return _text_column; }

  const auto& labelColumn() const { return _label_column; }

  auto labelDelimiter() const { return _label_delimiter; }

  proto::udt::TextDatasetConfig* toProto() const;

 private:
  std::string _text_column;
  std::string _label_column;
  std::optional<char> _label_delimiter;
};

class Featurizer {
 public:
  Featurizer(data::ColumnDataTypes data_types,
             const data::TemporalRelationships& temporal_relationships,
             const std::string& label_column,
             thirdai::data::TransformationPtr label_transform,
             thirdai::data::OutputColumnsList bolt_label_columns,
             const data::TabularOptions& options);

  explicit Featurizer(const proto::udt::Featurizer& featurizer);

  thirdai::data::LoaderPtr getDataLoader(
      const dataset::DataSourcePtr& data_source, size_t batch_size,
      bool shuffle, bool verbose,
      dataset::DatasetShuffleConfig shuffle_config =
          dataset::DatasetShuffleConfig());

  thirdai::data::LoaderPtr getColdStartDataLoader(
      const dataset::DataSourcePtr& data_source,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      bool fast_approximation, size_t batch_size, bool shuffle, bool verbose,
      dataset::DatasetShuffleConfig shuffle_config =
          dataset::DatasetShuffleConfig());

  bolt::TensorList featurizeInput(const MapInput& sample);

  bolt::TensorList featurizeInputBatch(const MapInputBatch& samples);

  bolt::TensorList featurizeInputColdStart(
      MapInput sample, const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names);

  std::pair<bolt::TensorList, bolt::TensorList> featurizeTrainingBatch(
      const MapInputBatch& samples);

  thirdai::data::ExplanationMap explain(
      const thirdai::data::ColumnMap& columns) {
    return _input_transform_non_updating->explain(columns, *_state);
  }

  const auto& state() const { return _state; }

  void updateTemporalTrackers(const MapInput& sample);

  void updateTemporalTrackersBatch(const MapInputBatch& samples);

  bool hasTemporalTransformations() const;

  const auto& textDatasetConfig() const {
    if (!_text_dataset) {
      throw std::runtime_error(
          "This method is only supported for models with a text input and a "
          "categorical output.");
    }
    return *_text_dataset;
  }

  void resetTemporalTrackers();

  proto::udt::Featurizer* toProto() const;

 protected:
  thirdai::data::LoaderPtr getDataLoaderHelper(
      const dataset::DataSourcePtr& data_source, size_t batch_size,
      bool shuffle, bool verbose, dataset::DatasetShuffleConfig shuffle_config,
      const thirdai::data::TransformationPtr& cold_start_transform = nullptr);

  thirdai::data::TransformationPtr coldStartTransform(
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      bool fast_approximation = false);

  thirdai::data::TransformationPtr _input_transform;
  thirdai::data::TransformationPtr _input_transform_non_updating;
  thirdai::data::TransformationPtr _label_transform;

  thirdai::data::OutputColumnsList _bolt_input_columns;
  thirdai::data::OutputColumnsList _bolt_label_columns;

  char _delimiter;

  thirdai::data::StatePtr _state;

  std::optional<TextDatasetConfig> _text_dataset;

  Featurizer() {}  // For cereal

 private:
  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive);
};

using FeaturizerPtr = std::shared_ptr<Featurizer>;

}  // namespace thirdai::automl