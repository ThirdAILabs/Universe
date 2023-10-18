#pragma once

#include "Featurizer.h"
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/rca/ExplanationMap.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::automl {

namespace feat = thirdai::data;

/**
 * We currently assume integer targets for simplicity.
 */
class UDTTransformationFactory {
 public:
  UDTTransformationFactory(
      ColumnDataTypes data_types,
      const UserProvidedTemporalRelationships& user_temporal_relationships,
      const std::string& label_column, const TabularOptions& options);

  static auto make(
      ColumnDataTypes data_types,
      const UserProvidedTemporalRelationships& user_temporal_relationships,
      const std::string& label_column, const TabularOptions& options) {
    return std::make_shared<UDTTransformationFactory>(
        std::move(data_types), user_temporal_relationships, label_column,
        options);
  }

  thirdai::data::TransformationPtr labeledTransformNoTemporalUpdates() {
    return data::Pipeline::make(
        {_input_transform_no_temporal_updates, _label_transform});
  }

  thirdai::data::TransformationPtr labeledTransformWithTemporalUpdates() const {
    return data::Pipeline::make(
        {_input_transform_with_temporal_updates, _label_transform});
  }

  thirdai::data::TransformationPtr inputTransformNoTemporalUpdates() const {
    return _input_transform_no_temporal_updates;
  }

  thirdai::data::TransformationPtr inputTransformWithTemporalUpdates() const {
    return _input_transform_with_temporal_updates;
  }

  const thirdai::data::TransformationPtr& labelTransform() const {
    return _label_transform;
  }

  const thirdai::data::OutputColumnsList& inputColumns() const {
    return _input_columns;
  }

  const std::string& labelColumn() const { return _label_column; }

  thirdai::data::TransformationPtr coldstartAugmentation(
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      bool fast_approximation) const;

  const auto& textDatasetConfig() const {
    if (!_text_dataset) {
      throw std::runtime_error(
          "This method is only supported for models with a text input and a "
          "categorical output.");
    }
    return *_text_dataset;
  }

  bool hasTemporalTransform() const { return _has_temporal_transform; }

 protected:
  feat::TransformationPtr _input_transform_with_temporal_updates;
  feat::TransformationPtr _input_transform_no_temporal_updates;
  feat::TransformationPtr _label_transform;
  feat::OutputColumnsList _input_columns;
  const std::string _label_column;

  std::optional<TextDatasetConfig> _text_dataset;
  bool _has_temporal_transform;

  UDTTransformationFactory() {}  // For cereal

 private:
  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive);
};

using UDTTransformationFactoryPtr = std::shared_ptr<UDTTransformationFactory>;

}  // namespace thirdai::automl