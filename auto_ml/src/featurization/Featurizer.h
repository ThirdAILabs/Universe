#pragma once

#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/rca/ExplanationMap.h>
#include <data/src/transformations/SpladeAugmentation.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/cold_start/VariableLengthColdStart.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <stdexcept>

namespace thirdai::automl {

class TextDatasetConfig {
 public:
  TextDatasetConfig(std::string text_column, std::string label_column,
                    std::optional<char> label_delimiter)
      : _text_column(std::move(text_column)),
        _label_column(std::move(label_column)),
        _label_delimiter(label_delimiter) {}

  const auto& textColumn() const { return _text_column; }

  const auto& labelColumn() const { return _label_column; }

  auto labelDelimiter() const { return _label_delimiter; }

  TextDatasetConfig() {}  // For cereal

 private:
  std::string _text_column;
  std::string _label_column;
  std::optional<char> _label_delimiter;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_text_column, _label_column, _label_delimiter);
  }
};

class Featurizer {
 public:
  Featurizer(ColumnDataTypes data_types,
             const TemporalRelationships& temporal_relationships,
             const std::string& label_column,
             data::TransformationPtr label_transform,
             data::OutputColumnsList bolt_label_columns,
             const TabularOptions& options);

  Featurizer(data::TransformationPtr input_transform,
             data::TransformationPtr const_input_transform,
             data::TransformationPtr label_transform,
             data::OutputColumnsList bolt_input_columns,
             data::OutputColumnsList bolt_label_columns, char delimiter,
             data::StatePtr state,
             std::optional<TextDatasetConfig> text_dataset);

  explicit Featurizer(const ar::Archive& archive);

  data::LoaderPtr getDataLoader(
      const dataset::DataSourcePtr& data_source, size_t batch_size,
      bool shuffle, bool verbose,
      const std::optional<data::SpladeConfig>& splade_config = std::nullopt,
      dataset::DatasetShuffleConfig shuffle_config =
          dataset::DatasetShuffleConfig());

  data::LoaderPtr getColdStartDataLoader(
      const dataset::DataSourcePtr& data_source,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      std::optional<data::VariableLengthConfig> variable_length,
      const std::optional<data::SpladeConfig>& splade_config,
      bool fast_approximation, size_t batch_size, bool shuffle, bool verbose,
      dataset::DatasetShuffleConfig shuffle_config =
          dataset::DatasetShuffleConfig());

  bolt::TensorList featurizeInput(const MapInput& sample, const std::optional<data::SpladeConfig>& splade_config=std::nullopt);

  bolt::TensorList featurizeInputBatch(const MapInputBatch& sampless, const std::optional<data::SpladeConfig>& splade_config=std::nullopt);

  bolt::TensorList featurizeInputColdStart(
      MapInput sample, const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names);

  std::pair<bolt::TensorList, bolt::TensorList> featurizeTrainingBatch(
      const MapInputBatch& samples);

  data::ExplanationMap explain(const data::ColumnMap& columns) {
    return _const_input_transform->explain(columns, *_state);
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

  ar::ConstArchivePtr toArchive() const;

  static std::shared_ptr<Featurizer> fromArchive(const ar::Archive& archive);

 protected:
  data::LoaderPtr getDataLoaderHelper(
      const dataset::DataSourcePtr& data_source, size_t batch_size,
      bool shuffle, bool verbose, dataset::DatasetShuffleConfig shuffle_config,
      const data::TransformationPtr& preprocesser = nullptr);

  data::TransformationPtr coldStartTransform(
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      std::optional<data::VariableLengthConfig> variable_length,
      bool fast_approximation = false);

  std::shared_ptr<ar::Map> toArchiveMap() const;

  data::TransformationPtr _input_transform;
  data::TransformationPtr _const_input_transform;
  data::TransformationPtr _label_transform;

  data::OutputColumnsList _bolt_input_columns;
  data::OutputColumnsList _bolt_label_columns;

  char _delimiter;

  data::StatePtr _state;

  std::optional<TextDatasetConfig> _text_dataset;

  Featurizer() {}  // For cereal

 private:
  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive);
};

bool hasTemporalTransformation(const data::TransformationPtr& t);

using FeaturizerPtr = std::shared_ptr<Featurizer>;

}  // namespace thirdai::automl