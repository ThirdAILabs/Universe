#pragma once

#include <cereal/access.hpp>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <auto_ml/src/featurization/Featurizer.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/TextCompat.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/mach/MachIndex.h>

namespace thirdai::automl {

using RlhfSample = std::pair<std::string, std::vector<uint32_t>>;

class MachFeaturizer final : public Featurizer {
 public:
  MachFeaturizer(ColumnDataTypes data_types,
                 const TemporalRelationships& temporal_relationship,
                 const std::string& label_column,
                 const dataset::mach::MachIndexPtr& mach_index,
                 const TabularOptions& options,
                 data::ValueFillType label_value_fill);

  MachFeaturizer(
      const std::shared_ptr<data::TextCompat>& text_transform,
      data::OutputColumnsList bolt_input_columns,
      const std::string& label_column,
      const dataset::mach::MachIndexPtr& mach_index, char csv_delimiter,
      std::optional<char> label_delimiter,
      data::ValueFillType label_value_fill = data::ValueFillType::Ones);

  explicit MachFeaturizer(const ar::Archive& archive);

  void insertNewDocIds(const dataset::DataSourcePtr& data_source);

  void insertNewDocIds(const data::ColumnMap& data);

  std::vector<std::pair<bolt::TensorList, std::vector<uint32_t>>>
  featurizeForIntroduceDocuments(
      const dataset::DataSourcePtr& data_source,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      bool fast_approximation, size_t batch_size);

  std::pair<bolt::TensorList, bolt::TensorList> featurizeTrainWithHashesBatch(
      const MapInputBatch& samples);

  data::ColumnMap featurizeDataset(
      const dataset::DataSourcePtr& data_source,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      const std::optional<data::VariableLengthConfig>& variable_length);

  data::ColumnMap featurizeRlhfSamples(const std::vector<RlhfSample>& samples);

  bolt::LabeledDataset columnsToTensors(const data::ColumnMap& columns,
                                        size_t batch_size) const;

  data::ColumnMap getBalancingSamples(
      const dataset::DataSourcePtr& data_source,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      std::optional<data::VariableLengthConfig> variable_length,
      size_t n_balancing_samples);

  const auto& machIndex() const { return _state->machIndex(); }

  // This is a redefinition of the Featurizer::toArchive method instead of an
  // override because the original Featurizer class was not declared without any
  // virtual methods, and adding a virtual method now messes up cereal when it
  // tries to load this class. This is ok though becuase we only every interact
  // with the MachFeaturizer directly, we never through a pointer to a
  // featurizer, thus calling this method will invoke the
  // MachFeaturizer::toArchive method, not the Featurizer::toArchive method.
  ar::ConstArchivePtr toArchive() const;

  static std::shared_ptr<MachFeaturizer> fromArchive(
      const ar::Archive& archive);

 private:
  data::ColumnMap removeIntermediateColumns(const data::ColumnMap& columns);

  static data::TransformationPtr makeDocIdTransformation(
      const std::string& label_column_name,
      std::optional<char> label_delimiter);

  static data::TransformationPtr makeLabelTransformations(
      const std::string& label_column_name,
      std::optional<char> label_delimiter);

  // The Mach model takes in two labels, one for the buckets, and one containing
  // the doc ids which is used by the mach metrics. For some inputs, for
  // instance in trainWithHashes, we don't have the doc ids that the model is
  // expecting, this adds a dummy input for the doc ids so that we have the
  // number of labels the model is expecting.
  static void addDummyDocIds(data::ColumnMap& columns);

  data::TransformationPtr _doc_id_transform;
  data::TransformationPtr _prehashed_labels_transform;

  MachFeaturizer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using MachFeaturizerPtr = std::shared_ptr<MachFeaturizer>;

}  // namespace thirdai::automl
