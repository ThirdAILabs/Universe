#include "ColdStartUtils.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <new_dataset/src/featurization_pipeline/augmentations/ColdStartText.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>

namespace thirdai::automl::cold_start {

void ColdStartMetaData::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void ColdStartMetaData::save_stream(std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

ColdStartMetaDataPtr ColdStartMetaData::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

ColdStartMetaDataPtr ColdStartMetaData::load_stream(
    std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<ColdStartMetaData> deserialize_into(new ColdStartMetaData());
  iarchive(*deserialize_into);
  return deserialize_into;
}

template <class Archive>
void ColdStartMetaData::serialize(Archive& archive) {
  archive(_label_delimiter, _label_column_name, _integer_target);
}

dataset::cold_start::ColdStartDataSourcePtr preprocessColdStartTrainSource(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    data::TabularDatasetFactoryPtr& dataset_factory,
    ColdStartMetaDataPtr &metadata) {
  if (!metadata->integerTarget()) {
    throw std::invalid_argument(
        "Cold start pretraining currently only supports integer labels.");
  }

  if (dataset_factory->inputDataTypes().size() != 1 ||
      !data::asText(dataset_factory->inputDataTypes().begin()->second)) {
    throw std::invalid_argument(
        "Cold start pretraining can only be used on datasets with a single "
        "text input column and target column. The current model is configured "
        "with " +
        std::to_string(dataset_factory->inputDataTypes().size()) +
        " input columns.");
  }

  std::string text_column_name =
      dataset_factory->inputDataTypes().begin()->first;

  auto dataset = thirdai::data::ColumnMap::createStringColumnMapFromFile(
      data, dataset_factory->delimiter());

  thirdai::data::ColdStartTextAugmentation augmentation(
      /* strong_column_names= */ strong_column_names,
      /* weak_column_names= */ weak_column_names,
      /* label_column_name= */ metadata->getLabelColumn(),
      /* output_column_name= */ text_column_name);

  auto augmented_data = augmentation.apply(dataset);

  auto data_source = thirdai::dataset::cold_start::ColdStartDataSource::make(
      /* column_map= */ augmented_data,
      /* text_column_name= */ text_column_name,
      /* label_column_name= */ metadata->getLabelColumn(),
      /* column_delimiter= */ dataset_factory->delimiter(),
      /* label_delimiter= */ metadata->getLabelDelimiter(),
      /* resource_name = */ data->resourceName());

  return data_source;
}
}  // namespace thirdai::automl::cold_start