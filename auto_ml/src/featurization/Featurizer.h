#pragma once

#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/rca/ExplanationMap.h>
#include <data/src/transformations/State.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <stdexcept>

namespace thirdai::automl {

class TextDatasetConfig {
 public:
  TextDatasetConfig(std::string text_column, std::string label_column)
      : _text_column(std::move(text_column)),
        _label_column(std::move(label_column)) {}

  const auto& textColumn() const { return _text_column; }

  const auto& labelColumn() const { return _label_column; }

 private:
  std::string _text_column;
  std::string _label_column;
};

class Featurizer {
 public:
  Featurizer(data::ColumnDataTypes data_types,
             const data::TemporalRelationships& temporal_relationships,
             const std::string& label_column,
             thirdai::data::TransformationPtr label_transform,
             thirdai::data::IndexValueColumnList bolt_label_columns,
             const data::TabularOptions& options);

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
    return _input_transform->explain(columns, *_state);
  }

  const auto& state() const { return _state; }

  bool hasTemporalTransformations() const;

  const auto& textDatasetConfig() const {
    if (!_text_dataset) {
      throw std::runtime_error(
          "This method is only supported for text models.");
    }
    return *_text_dataset;
  }

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

  thirdai::data::IndexValueColumnList _bolt_input_columns;
  thirdai::data::IndexValueColumnList _bolt_label_columns;

  char _delimiter;

  thirdai::data::StatePtr _state;

  std::optional<TextDatasetConfig> _text_dataset;

 private:
  Featurizer() {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_input_transform, _input_transform_non_updating, _label_transform,
            _bolt_input_columns, _bolt_label_columns, _delimiter);
    // TODO(Nicholas) finish this
  }
};

using FeaturizerPtr = std::shared_ptr<Featurizer>;

}  // namespace thirdai::automl