#pragma once

#include <cereal/access.hpp>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <auto_ml/src/featurization/Featurizer.h>
#include <data/src/TensorConversion.h>
#include <dataset/src/mach/MachIndex.h>

namespace thirdai::automl {

using RlhfSample = std::pair<std::string, std::vector<uint32_t>>;

class MachFeaturizer final : public Featurizer {
 public:
  MachFeaturizer(data::ColumnDataTypes data_types,
                 const data::TemporalRelationships& temporal_relationship,
                 const std::string& label_column,
                 const dataset::mach::MachIndexPtr& mach_index,
                 const data::TabularOptions& options);

  explicit MachFeaturizer(const proto::udt::MachFeaturizer& featurizer);

  std::vector<std::pair<bolt::TensorList, std::vector<uint32_t>>>
  featurizeForIntroduceDocuments(
      const dataset::DataSourcePtr& data_source,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      bool fast_approximation, size_t batch_size);

  std::pair<bolt::TensorList, bolt::TensorList> featurizeHashesTrainingBatch(
      const MapInputBatch& samples);

  thirdai::data::ColumnMap featurizeDataset(
      const dataset::DataSourcePtr& data_source,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names);

  thirdai::data::ColumnMap featurizeRlhfSamples(
      const std::vector<RlhfSample>& samples);

  bolt::LabeledDataset columnsToTensors(const thirdai::data::ColumnMap& columns,
                                        size_t batch_size) const;

  std::vector<std::pair<uint32_t, RlhfSample>> getBalancingSamples(
      const dataset::DataSourcePtr& data_source,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      size_t n_balancing_samples, size_t rows_to_read);

  const auto& machIndex() const { return _state->machIndex(); }

  proto::udt::MachFeaturizer* toProto() const;

 private:
  thirdai::data::ColumnMap removeIntermediateColumns(
      const thirdai::data::ColumnMap& columns);

  static thirdai::data::TransformationPtr makeDocIdTransformation(
      const std::string& label_column_name,
      const data::CategoricalDataTypePtr& label_column_info);

  static thirdai::data::TransformationPtr makeLabelTransformations(
      const std::string& label_column_name,
      const data::CategoricalDataTypePtr& label_column_info);

  static void addDummyDocIds(thirdai::data::ColumnMap& columns);

  thirdai::data::TransformationPtr _doc_id_transform;
  thirdai::data::TransformationPtr _prehashed_labels_transform;

  MachFeaturizer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using MachFeaturizerPtr = std::shared_ptr<MachFeaturizer>;

}  // namespace thirdai::automl
