#pragma once

#include <cereal/access.hpp>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/featurization/Featurizer.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/mach/MachIndex.h>
#include <algorithm>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace thirdai::automl {

using RlhfSample = std::pair<std::string, std::vector<uint32_t>>;

class NewMachFeaturizer final : public Featurizer {
 public:
  NewMachFeaturizer(ColumnDataTypes data_types,
                    const TemporalRelationships& temporal_relationship,
                    const std::string& label_column,
                    const TabularOptions& options);

  data::ColumnMapIteratorPtr iter(const dataset::DataSourcePtr& data) const {
    return data::CsvIterator::make(data, _delimiter);
  }

  data::ColumnMap columns(const dataset::DataSourcePtr& data) const {
    return data::CsvIterator::all(data, _delimiter);
  }

  data::ColumnMap addLabelColumn(data::ColumnMap&& columns,
                                 uint32_t label) const {
    data::ColumnPtr label_column;
    if (_label_delimiter) {
      std::vector<std::vector<uint32_t>> label_column_data(columns.numRows(),
                                                           {label});
      label_column = data::ArrayColumn<uint32_t>::make(
          std::move(label_column_data), std::numeric_limits<uint32_t>::max());
    } else {
      std::vector<uint32_t> label_column_data(columns.numRows(), label);
      label_column = data::ValueColumn<uint32_t>::make(
          std::move(label_column_data), std::numeric_limits<uint32_t>::max());
    }
    columns.setColumn(_label_column, label_column);
    return columns;
  }

  std::pair<data::ColumnMap, data::ColumnMap> associationColumnMaps(
      const std::vector<std::pair<std::string, std::string>>& samples) const;

  data::ColumnMap upvoteColumnMap(
      const std::vector<std::pair<std::string, uint32_t>>& samples) const;

  data::ColumnMapIteratorPtr trackingLabeledTransform(
      data::ColumnMapIteratorPtr&& iter) const {
    return data::TransformedIterator::make(
        iter, /* transformation= */ labeledTransformWithTrackerUpdates(),
        /* state= */ _state);
  }

  data::ColumnMap trackingLabeledTransform(data::ColumnMap&& columns) const {
    return labeledTransformWithTrackerUpdates()->apply(std::move(columns),
                                                       *_state);
  }

  data::ColumnMapIteratorPtr constLabeledTransform(
      data::ColumnMapIteratorPtr&& iter) const {
    return data::TransformedIterator::make(
        iter, /* transformation= */ labeledTransformConst(),
        /* state= */ _state);
  }

  data::ColumnMap constLabeledTransform(data::ColumnMap&& columns) const {
    return labeledTransformConst()->apply(std::move(columns), *_state);
  }

  data::ColumnMapIteratorPtr constBucketedTransform(
      data::ColumnMapIteratorPtr&& iter) const {
    return data::TransformedIterator::make(
        iter, /* transformation= */ bucketedTransformConst(),
        /* state= */ _state);
  }

  data::ColumnMap constBucketedTransform(data::ColumnMap&& columns) const {
    return bucketedTransformConst()->apply(std::move(columns), *_state);
  }

  data::ColumnMapIteratorPtr constUnlabeledTransform(
      data::ColumnMapIteratorPtr&& iter) const {
    return data::TransformedIterator::make(
        iter, /* transformation= */ unlabeledTransformConst(),
        /* state= */ _state);
  }

  data::ColumnMap constUnlabeledTransform(data::ColumnMap&& columns) const {
    return unlabeledTransformConst()->apply(std::move(columns), *_state);
  }

  data::ColumnMapIteratorPtr coldstart(
      data::ColumnMapIteratorPtr&& iter,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names) const {
    return data::TransformedIterator::make(
        iter, /* transformation= */
        coldstartTransform(strong_column_names, weak_column_names),
        /* state= */ _state);
  }

  data::ColumnMap coldstart(
      data::ColumnMap&& columns,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names) const {
    return coldstartTransform(strong_column_names, weak_column_names)
        ->applyStateless(std::move(columns));
  }

  data::ColumnMap concatTextColumns(
      data::ColumnMap&& columns,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names) const;

  void assertNoTemporalFeatures() const;

  data::TransformationPtr labeledTransformWithTrackerUpdates() const;

  data::TransformationPtr labeledTransformConst() const;

  data::TransformationPtr unlabeledTransformConst() const;

  data::TransformationPtr bucketedTransformConst() const;

  data::TransformationPtr coldstartTransform(
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names) const;

 private:
  NewMachFeaturizer() {}

  char _delimiter;
  std::optional<char> _label_delimiter;
  std::string _label_column;
  data::StatePtr _state;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using NewMachFeaturizerPtr = std::shared_ptr<NewMachFeaturizer>;

}  // namespace thirdai::automl
