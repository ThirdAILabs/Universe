#pragma once

#include <cereal/access.hpp>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/featurization/Featurizer.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/featurization/TabularTransformations.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/CategoricalTemporal.h>
#include <data/src/transformations/ColdStartText.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringConcat.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/mach/MachIndex.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace thirdai::automl {

class MachFeaturizer final {
 public:
  MachFeaturizer(ColumnDataTypes data_types,
                 const CategoricalDataTypePtr& target_config,
                 const TemporalRelationships& temporal_relationships,
                 const std::string& label_column,
                 const TabularOptions& options);

  static auto make(ColumnDataTypes data_types,
                   const CategoricalDataTypePtr& target_config,
                   const TemporalRelationships& temporal_relationships,
                   const std::string& label_column,
                   const TabularOptions& options) {
    return std::make_shared<MachFeaturizer>(
        std::move(data_types), target_config, temporal_relationships,
        label_column, options);
  }

  data::ColumnMapIteratorPtr iter(const dataset::DataSourcePtr& data) const {
    return data::CsvIterator::make(data, _delimiter);
  }

  data::ColumnMap columns(const dataset::DataSourcePtr& data) const {
    return data::CsvIterator::all(data, _delimiter);
  }

  data::ColumnMap addLabelColumn(data::ColumnMap&& columns,
                                 uint32_t label) const;

  std::pair<data::ColumnMap, data::ColumnMap> associationColumnMaps(
      const std::vector<std::pair<std::string, std::string>>& samples) const;

  data::ColumnMap upvoteLabeledColumnMap(
      const std::vector<std::pair<std::string, uint32_t>>& samples) const;

  data::ColumnMapIteratorPtr labeledTransform(
      data::ColumnMapIteratorPtr&& iter) const {
    return data::TransformedIterator::make(
        iter, trackingLabeledTransformation(), _state);
  }

  data::ColumnMap labeledTransform(data::ColumnMap&& columns) const {
    return trackingLabeledTransformation()->apply(std::move(columns), *_state);
  }

  data::ColumnMapIteratorPtr bucketedTransform(
      data::ColumnMapIteratorPtr&& iter) const {
    return data::TransformedIterator::make(
        iter, trackingBucketedTransformation(), _state);
  }

  data::ColumnMap bucketedTransform(data::ColumnMap&& columns) const {
    return trackingBucketedTransformation()->apply(std::move(columns), *_state);
  }

  data::ColumnMapIteratorPtr trackingUnlabeledTransform(
      data::ColumnMapIteratorPtr&& iter) const {
    return data::TransformedIterator::make(iter, _tracking_input_transformation,
                                           _state);
  }

  data::ColumnMap trackingUnlabeledTransform(data::ColumnMap&& columns) const {
    return _tracking_input_transformation->apply(std::move(columns), *_state);
  }

  data::ColumnMapIteratorPtr constUnlabeledTransform(
      data::ColumnMapIteratorPtr&& iter) const {
    return data::TransformedIterator::make(iter, _const_input_transformation,
                                           _state);
  }

  data::ColumnMap constUnlabeledTransform(data::ColumnMap&& columns) const {
    return _const_input_transformation->apply(std::move(columns), *_state);
  }

  data::ColumnMapIteratorPtr coldstart(
      data::ColumnMapIteratorPtr&& iter,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      bool fast_approximation = false) const {
    return data::TransformedIterator::make(
        iter,
        coldstartTransformation(strong_column_names, weak_column_names,
                                fast_approximation),
        _state);
  }

  data::ColumnMap coldstart(data::ColumnMap&& columns,
                            const std::vector<std::string>& strong_column_names,
                            const std::vector<std::string>& weak_column_names,
                            bool fast_approximation = false) const {
    return coldstartTransformation(strong_column_names, weak_column_names,
                                   fast_approximation)
        ->applyStateless(std::move(columns));
  }

  void assertNoTemporalFeatures() const {
    if (hasTemporalTransformations()) {
      throw std::runtime_error(
          "This feature is not supported for models with temporal features.");
    }
  }

  void assertTextModel() const {
    if (!_text_dataset_config) {
      throw std::runtime_error(
          "This feature is only supported for text models.");
    }
  }

  bool hasTemporalTransformations() const;

  void resetTemporalTrackers() { _state->clearHistoryTrackers(); }

  const TextDatasetConfig& textDatasetConfig() const {
    assertTextModel();
    return *_text_dataset_config;
  }

 private:
  data::TransformationPtr trackingLabeledTransformation() const {
    return data::Pipeline::make(
        {_tracking_input_transformation, _label_transformation});
  }

  data::TransformationPtr trackingBucketedTransformation() const {
    return data::Pipeline::make(
        {_tracking_input_transformation, _string_to_int_buckets});
  }

  data::TransformationPtr coldstartTransformation(
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      bool fast_approximation) const;

  MachFeaturizer() {}

  char _delimiter;
  std::optional<char> _label_delimiter;
  data::StatePtr _state;

  data::TransformationPtr _tracking_input_transformation;
  std::string _input_indices_column;
  std::string _input_values_column;
  data::TransformationPtr _const_input_transformation;
  data::TransformationPtr _label_transformation;
  data::TransformationPtr _string_to_int_buckets;

  std::optional<TextDatasetConfig> _text_dataset_config;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using MachFeaturizerPtr = std::shared_ptr<MachFeaturizer>;

}  // namespace thirdai::automl
