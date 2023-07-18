#include "ColdStartUtils.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/ColdStartText.h>
#include <dataset/src/DataSource.h>

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
  archive(_label_delimiter, _label_column_name);
}

void verifyDataTypes(data::TabularDatasetFactoryPtr& dataset_factory) {
  if (dataset_factory->inputDataTypes().size() != 1 ||
      !data::asText(dataset_factory->inputDataTypes().begin()->second)) {
    throw std::invalid_argument(
        "This function can only be used on datasets with a single "
        "text input column and target column. The current model is configured "
        "with " +
        std::to_string(dataset_factory->inputDataTypes().size()) +
        " input columns.");
  }
}

dataset::cold_start::ColdStartDataSourcePtr preprocessColdStartTrainSource(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    data::TabularDatasetFactoryPtr& dataset_factory,
    ColdStartMetaDataPtr& metadata) {
  verifyDataTypes(dataset_factory);
  std::string text_column_name =
      dataset_factory->inputDataTypes().begin()->first;

  auto csv_data_source =
      dataset::CsvDataSource::make(data, dataset_factory->delimiter());

  auto dataset = thirdai::data::ColumnMap::createStringColumnMapFromFile(
      csv_data_source, dataset_factory->delimiter());

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

dataset::cold_start::ColdStartDataSourcePtr concatenatedDocumentDataSource(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    data::TabularDatasetFactoryPtr& dataset_factory,
    ColdStartMetaDataPtr& metadata) {
  verifyDataTypes(dataset_factory);

  std::string text_column_name =
      dataset_factory->inputDataTypes().begin()->first;

  auto csv_data_source =
      dataset::CsvDataSource::make(data, dataset_factory->delimiter());

  auto dataset = thirdai::data::ColumnMap::createStringColumnMapFromFile(
      csv_data_source, dataset_factory->delimiter());

  std::vector<std::string> column_names = weak_column_names;
  column_names.insert(column_names.end(), strong_column_names.begin(),
                      strong_column_names.end());

  std::vector<std::string> samples;
  auto label_column =
      dataset.getValueColumn<std::string>(metadata->getLabelColumn());
  for (uint64_t row_id = 0; row_id < label_column->numRows(); row_id++) {
    std::string output_sample;
    for (const auto& column_name : column_names) {
      auto column = dataset.getValueColumn<std::string>(column_name);
      output_sample.append(column->value(row_id));
      output_sample.append(" ");
    }
    samples.push_back(output_sample);
  }

  thirdai::data::ValueColumnPtr<std::string> augmented_data_column =
      thirdai::data::StringColumn::make(std::move(samples));

  std::unordered_map<std::string, thirdai::data::ColumnPtr> new_columns;
  new_columns.emplace(metadata->getLabelColumn(), label_column);
  new_columns.emplace(text_column_name, augmented_data_column);
  thirdai::data::ColumnMap new_column_map(new_columns);

  auto data_source = thirdai::dataset::cold_start::ColdStartDataSource::make(
      /* column_map= */ new_column_map,
      /* text_column_name= */ text_column_name,
      /* label_column_name= */ metadata->getLabelColumn(),
      /* column_delimiter= */ dataset_factory->delimiter(),
      /* label_delimiter= */ metadata->getLabelDelimiter(),
      /* resource_name = */ data->resourceName());

  return data_source;
}

}  // namespace thirdai::automl::cold_start