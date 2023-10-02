#pragma once

#include "Featurizer.h"
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/rca/ExplanationMap.h>
#include <data/src/transformations/AddBalancingSamples.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace thirdai::automl {

namespace feat = thirdai::data;

/**
 * We currently assume integer targets for simplicity.
 */
class UDTTransformationFactory {
 public:
  UDTTransformationFactory(data::ColumnDataTypes data_types,
                           const data::UserProvidedTemporalRelationships&
                               user_temporal_relationships,
                           const std::string& label_column,
                           thirdai::data::ValueFillType label_value_fill,
                           const data::TabularOptions& options);

  static auto make(data::ColumnDataTypes data_types,
                   const data::UserProvidedTemporalRelationships&
                       user_temporal_relationships,
                   const std::string& label_column,
                   thirdai::data::ValueFillType label_value_fill,
                   const data::TabularOptions& options) {
    return std::make_shared<UDTTransformationFactory>(
        std::move(data_types), user_temporal_relationships, label_column,
        label_value_fill, options);
  }

  const thirdai::data::TransformationPtr& trainInputTransform() const {
    return _train_input_transform;
  }

  const thirdai::data::TransformationPtr& inferInputTransform() const {
    return _infer_input_transform;
  }

  const thirdai::data::TransformationPtr& labelTransform() const {
    return _label_transform;
  }

  const thirdai::data::OutputColumnsList& inputColumns() const {
    return _input_columns;
  }

  const thirdai::data::OutputColumnsList& labelColumns() const {
    return _label_columns;
  }

  thirdai::data::TransformationPtr unsupAugmenter(
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      bool fast_approximation) const;

  thirdai::data::TransformationPtr storeBalancers() const {
    return thirdai::data::AddBalancingSamples::make(
        _text_dataset->textColumn(), _text_dataset->labelColumn());
  }

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
  feat::TransformationPtr _train_input_transform;
  feat::TransformationPtr _infer_input_transform;
  feat::TransformationPtr _label_transform;
  feat::OutputColumnsList _input_columns;
  feat::OutputColumnsList _label_columns;

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