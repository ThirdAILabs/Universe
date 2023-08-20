#include "MachFeaturizer.h"
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

static const std::string PARSED_DOC_ID_COLUMN = "__doc_ids__";
static const std::string MACH_LABEL_COLUMN = "__mach_labels__";

MachFeaturizer::MachFeaturizer(
    data::ColumnDataTypes data_types,
    const data::TemporalRelationships& temporal_relationship,
    const std::string& label_column, dataset::mach::MachIndexPtr mach_index,
    const data::TabularOptions& options)
    : Featurizer(
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
MachFeaturizer::featurizeForIntroduceDocuments(
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
MachFeaturizer::featurizeHashesTrainingBatch(const MapInputBatch& samples) {
  auto columns = thirdai::data::ColumnMap::fromMapInputBatch(samples);

  columns = _input_transform->apply(columns, *_state);
  columns = _prehashed_labels_transform->apply(columns, *_state);

  addDummyDocIds(columns);

  auto data = thirdai::data::toTensors(columns, _bolt_input_columns);
  auto labels = thirdai::data::toTensors(columns, _bolt_label_columns);

  return std::make_pair(std::move(data), std::move(labels));
}

thirdai::data::ColumnMap MachFeaturizer::featurizeDataset(
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

  // Remove intermediate columns.
  thirdai::data::ColumnMap output({});
  for (const auto& [index_col, value_col] : _bolt_input_columns) {
    output.setColumn(index_col, columns.getColumn(index_col));
    if (value_col) {
      output.setColumn(*value_col, columns.getColumn(*value_col));
    }
  }
  for (const auto& [index_col, value_col] : _bolt_label_columns) {
    output.setColumn(index_col, columns.getColumn(index_col));
    if (value_col) {
      output.setColumn(*value_col, columns.getColumn(*value_col));
    }
  }

  return output;
}

thirdai::data::ColumnMap MachFeaturizer::featurizeRlhfSamples(
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
      MACH_LABEL_COLUMN,
      thirdai::data::ArrayColumn<uint32_t>::make(std::move(labels)));

  addDummyDocIds(columns);

  return columns;
}

bolt::LabeledDataset MachFeaturizer::columnsToTensors(
    const thirdai::data::ColumnMap& columns, size_t batch_size) const {
  auto data =
      thirdai::data::toTensorBatches(columns, _bolt_input_columns, batch_size);
  auto labels =
      thirdai::data::toTensorBatches(columns, _bolt_label_columns, batch_size);

  return std::make_pair(std::move(data), std::move(labels));
}

std::vector<std::pair<uint32_t, RlhfSample>>
MachFeaturizer::getBalancingSamples(
    const dataset::DataSourcePtr& data_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    size_t n_balancing_samples, size_t rows_to_read) {
  thirdai::data::ColumnMapIterator data_iter(
      data_source, _delimiter, std::max(n_balancing_samples, rows_to_read));

  auto columns = data_iter.next().value();

  if (!strong_column_names.empty() || !weak_column_names.empty()) {
    coldStartTransform(strong_column_names, weak_column_names)
        ->apply(columns, *_state);
  }

  _label_transform->apply(columns, *_state);

  columns.shuffle();

  auto text_col =
      columns.getValueColumn<std::string>(textDatasetConfig().textColumn());
  auto mach_label_col = columns.getArrayColumn<uint32_t>(MACH_LABEL_COLUMN);
  auto doc_id_col = columns.getValueColumn<uint32_t>(PARSED_DOC_ID_COLUMN);

  size_t n_to_return = std::min(n_balancing_samples, columns.numRows());
  std::vector<std::pair<uint32_t, RlhfSample>> samples;
  for (size_t i = 0; i < n_to_return; i++) {
    auto labels = mach_label_col->row(i);

    RlhfSample sample(text_col->value(i),
                      std::vector<uint32_t>(labels.begin(), labels.end()));

    samples.emplace_back(doc_id_col->value(i), std::move(sample));
  }

  return samples;
}

thirdai::data::TransformationPtr MachFeaturizer::makeLabelTransformations(
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

void MachFeaturizer::addDummyDocIds(thirdai::data::ColumnMap& columns) {
  auto dummy_doc_ids = thirdai::data::ValueColumn<uint32_t>::make(
      std::vector<uint32_t>(columns.numRows(), 0), std::nullopt);

  columns.setColumn(PARSED_DOC_ID_COLUMN, dummy_doc_ids);
}

}  // namespace thirdai::automl