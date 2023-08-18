#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <auto_ml/src/featurization/DatasetFactory.h>
#include <data/src/TensorConversion.h>
#include <dataset/src/mach/MachIndex.h>

namespace thirdai::automl {

using RlhfSample = std::pair<std::string, std::vector<uint32_t>>;

class MachDatasetFactory final : public DatasetFactory {
 public:
  MachDatasetFactory(data::ColumnDataTypes data_types,
                     const data::TemporalRelationships& temporal_relationship,
                     const std::string& label_column,
                     dataset::mach::MachIndexPtr mach_index,
                     const data::TabularOptions& options);

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

 private:
  thirdai::data::TransformationPtr makeLabelTransformations(
      const std::string& label_column_name,
      const data::CategoricalDataTypePtr& label_column_info,
      const dataset::mach::MachIndexPtr& mach_index);

  static void addDummyDocIds(thirdai::data::ColumnMap& columns);

  thirdai::data::TransformationPtr _doc_id_transform;
  thirdai::data::TransformationPtr _prehashed_labels_transform;

  static const std::string PARSED_DOC_ID_COLUMN;
  static const std::string MACH_LABEL_COLUMN;
};

}  // namespace thirdai::automl