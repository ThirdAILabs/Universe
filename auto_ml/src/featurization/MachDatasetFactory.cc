#include "MachDatasetFactory.h"
#include <_types/_uint32_t.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/TransformationList.h>
#include <dataset/src/mach/MachIndex.h>
#include <optional>
#include <stdexcept>

namespace thirdai::automl {

MachDatasetFactory::MachDatasetFactory(
    data::ColumnDataTypes data_types,
    const data::TemporalRelationships& temporal_relationship,
    const std::string& label_column, dataset::mach::MachIndexPtr mach_index,
    const data::TabularOptions& options)
    : DatasetFactory(
          data_types, temporal_relationship, label_column,
          makeLabelTransformations(
              label_column, data::asCategorical(data_types.at(label_column)),
              mach_index),
          {{MACH_LABEL_COLUMN, std::nullopt},
           {PARSED_DOC_ID_COLUMN, std::nullopt}},
          options) {
  _state = std::make_shared<thirdai::data::State>(mach_index);
}

std::vector<std::pair<bolt::TensorList, std::vector<uint32_t>>>
MachDatasetFactory::featurizeForIntroduceDocuments(
    const dataset::DataSourcePtr& data_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, bool fast_approximation,
    size_t batch_size) {
  auto csv_data_source = dataset::CsvDataSource::make(data_source, _delimiter);

  thirdai::data::ColumnMap columns =
      thirdai::data::ColumnMapIterator::all(data_source, _delimiter);

  auto transform = thirdai::data::TransformationList::make({
      coldStartTransform(strong_column_names, weak_column_names,
                         fast_approximation),
      _input_transform,
      _doc_id_transform,
  });

  columns = transform->apply(columns, *_state);

  auto input_tensors =
      thirdai::data::toTensorBatches(columns, _bolt_input_columns, batch_size);

  auto doc_ids = columns.getValueColumn<uint32_t>(PARSED_DOC_ID_COLUMN);

  std::vector<std::pair<bolt::TensorList, std::vector<uint32_t>>> batches;

  size_t row_idx = 0;
  for (const auto& tensor : input_tensors) {
    std::vector<uint32_t> batch_doc_ids(tensor.at(0)->batchSize());
    for (uint32_t& batch_doc_id : batch_doc_ids) {
      batch_doc_id = doc_ids->value(row_idx++);
    }
    batches.emplace_back(tensor, std::move(batch_doc_ids));
  }

  return batches;
}

std::pair<bolt::TensorList, bolt::TensorList>
MachDatasetFactory::featurizeHashesTrainingBatch(const MapInputBatch& samples) {
  auto columns = thirdai::data::ColumnMap::fromMapInputBatch(samples);

  columns = _input_transform->apply(columns, *_state);
  columns = _prehashed_labels_transform->apply(columns, *_state);

  addDummyDocIds(columns);

  auto data = thirdai::data::toTensors(columns, _bolt_input_columns);
  auto labels = thirdai::data::toTensors(columns, _bolt_label_columns);

  return std::make_pair(std::move(data), std::move(labels));
}

thirdai::data::ColumnMap MachDatasetFactory::featurizeDataset(
    const dataset::DataSourcePtr& data_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names) {
  thirdai::data::ColumnMap columns =
      thirdai::data::ColumnMapIterator::all(data_source, _delimiter);

  if (!strong_column_names.empty() || !weak_column_names.empty()) {
    coldStartTransform(strong_column_names, weak_column_names)
        ->apply(columns, *_state);
  }

  _input_transform->apply(columns, *_state);
  _label_transform->apply(columns, *_state);

  return columns;
}

thirdai::data::ColumnMap MachDatasetFactory::featurizeRlhfSamples(
    const std::vector<RlhfSample>& samples) {
  if (_text_dataset) {
    throw std::invalid_argument("RLHF is only supported for text datasets.");
  }

  std::vector<std::string> text;
  std::vector<std::vector<uint32_t>> labels;
  for (const auto& sample : samples) {
    text.push_back(sample.first);
    labels.push_back(sample.second);
  }

  thirdai::data::ColumnMap columns(
      {{_text_dataset->textColumn(),
        thirdai::data::ValueColumn<std::string>::make(std::move(text))}});

  _input_transform->apply(columns, *_state);

  columns.setColumn(
      _text_dataset->labelColumn(),
      thirdai::data::ArrayColumn<uint32_t>::make(std::move(labels)));

  addDummyDocIds(columns);

  return columns;
}

const std::string MachDatasetFactory::PARSED_DOC_ID_COLUMN = "__doc_ids__";
const std::string MachDatasetFactory::MACH_LABEL_COLUMN = "__mach_labels__";

thirdai::data::TransformationPtr MachDatasetFactory::makeLabelTransformations(
    const std::string& label_column_name,
    const data::CategoricalDataTypePtr& label_column_info,
    const dataset::mach::MachIndexPtr& mach_index) {
  if (auto delim = label_column_info->delimiter) {
    _doc_id_transform = std::make_shared<thirdai::data::StringToTokenArray>(
        label_column_name, PARSED_DOC_ID_COLUMN, delim, std::nullopt);
  } else {
    _doc_id_transform = std::make_shared<thirdai::data::StringToToken>(
        label_column_name, PARSED_DOC_ID_COLUMN, std::nullopt);
  }

  auto mach_label_transform = std::make_shared<thirdai::data::MachLabel>(
      PARSED_DOC_ID_COLUMN, MACH_LABEL_COLUMN);

  _prehashed_labels_transform =
      std::make_shared<thirdai::data::StringToTokenArray>(
          label_column_name, MACH_LABEL_COLUMN, ' ', mach_index->numBuckets());

  return thirdai::data::TransformationList::make(
      {_doc_id_transform, mach_label_transform});
}

void MachDatasetFactory::addDummyDocIds(thirdai::data::ColumnMap& columns) {
  auto dummy_doc_ids = thirdai::data::ValueColumn<uint32_t>::make(
      std::vector<uint32_t>(columns.numRows(), 0), std::nullopt);

  columns.setColumn(PARSED_DOC_ID_COLUMN, dummy_doc_ids);
}

}  // namespace thirdai::automl