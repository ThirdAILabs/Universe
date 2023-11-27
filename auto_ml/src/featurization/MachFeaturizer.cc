#include "MachFeaturizer.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/mach/MachIndex.h>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::automl {

MachFeaturizer::MachFeaturizer(
    ColumnDataTypes data_types,
    const TemporalRelationships& temporal_relationship,
    const std::string& label_column,
    const dataset::mach::MachIndexPtr& mach_index,
    const TabularOptions& options)
    : Featurizer(
          data_types, temporal_relationship, label_column,
          makeLabelTransformations(label_column,
                                   asCategorical(data_types.at(label_column))),
          {data::OutputColumns(MACH_LABELS), data::OutputColumns(MACH_DOC_IDS)},
          options) {
  _state = std::make_shared<data::State>(mach_index);

  _prehashed_labels_transform = std::make_shared<data::StringToTokenArray>(
      label_column, MACH_LABELS, ' ', mach_index->numBuckets());

  _doc_id_transform = makeDocIdTransformation(
      label_column, asCategorical(data_types.at(label_column)));
}

std::vector<std::pair<bolt::TensorList, std::vector<uint32_t>>>
MachFeaturizer::featurizeForIntroduceDocuments(
    const dataset::DataSourcePtr& data_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, bool fast_approximation,
    size_t batch_size) {
  auto csv_data_source = dataset::CsvDataSource::make(data_source, _delimiter);

  data::ColumnMap columns = data::CsvIterator::all(csv_data_source, _delimiter);

  auto transform = data::Pipeline::make({
      coldStartTransform(strong_column_names, weak_column_names,
                         fast_approximation),
      _input_transform,
      _doc_id_transform,
  });

  columns = transform->apply(columns, *_state);

  auto input_tensors =
      data::toTensorBatches(columns, _bolt_input_columns, batch_size);

  auto doc_ids = columns.getArrayColumn<uint32_t>(MACH_DOC_IDS);

  std::vector<std::pair<bolt::TensorList, std::vector<uint32_t>>> batches;

  size_t row_idx = 0;
  for (const auto& tensor : input_tensors) {
    std::vector<uint32_t> batch_doc_ids(tensor.at(0)->batchSize());
    for (uint32_t& batch_doc_id : batch_doc_ids) {
      auto row = doc_ids->row(row_idx++);
      // Each document in introduce documents should only have a single ID.
      if (row.size() != 1) {
        throw std::invalid_argument(
            "Expected only 1 document ID per row in introduceDocuments.");
      }
      batch_doc_id = row[0];
    }
    batches.emplace_back(tensor, std::move(batch_doc_ids));
  }

  return batches;
}

std::pair<bolt::TensorList, bolt::TensorList>
MachFeaturizer::featurizeHashesTrainingBatch(const MapInputBatch& samples) {
  auto columns = data::ColumnMap::fromMapInputBatch(samples);

  columns = _input_transform->apply(columns, *_state);
  columns = _prehashed_labels_transform->apply(columns, *_state);

  addDummyDocIds(columns);

  auto data = data::toTensors(columns, _bolt_input_columns);
  auto labels = data::toTensors(columns, _bolt_label_columns);

  return std::make_pair(std::move(data), std::move(labels));
}

data::ColumnMap MachFeaturizer::featurizeDataset(
    const dataset::DataSourcePtr& data_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names) {
  auto csv_data_source = dataset::CsvDataSource::make(data_source, _delimiter);

  data::ColumnMap columns = data::CsvIterator::all(csv_data_source, _delimiter);

  if (!strong_column_names.empty() || !weak_column_names.empty()) {
    columns = coldStartTransform(strong_column_names, weak_column_names)
                  ->apply(columns, *_state);
  }

  columns = _input_transform->apply(columns, *_state);
  columns = _label_transform->apply(columns, *_state);

  return removeIntermediateColumns(columns);
}

data::ColumnMap MachFeaturizer::featurizeRlhfSamples(
    const std::vector<RlhfSample>& samples) {
  if (!_text_dataset) {
    throw std::invalid_argument("RLHF is only supported for text datasets.");
  }

  std::vector<std::string> text;
  std::vector<std::vector<uint32_t>> labels;
  for (const auto& sample : samples) {
    text.push_back(sample.first);
    labels.push_back(sample.second);
  }

  data::ColumnMap columns(
      {{_text_dataset->textColumn(),
        data::ValueColumn<std::string>::make(std::move(text))}});

  columns = _input_transform->apply(columns, *_state);

  columns.setColumn(MACH_LABELS,
                    data::ArrayColumn<uint32_t>::make(
                        std::move(labels), _state->machIndex()->numBuckets()));

  addDummyDocIds(columns);

  return removeIntermediateColumns(columns);
}

bolt::LabeledDataset MachFeaturizer::columnsToTensors(
    const data::ColumnMap& columns, size_t batch_size) const {
  auto data = data::toTensorBatches(columns, _bolt_input_columns, batch_size);
  auto labels = data::toTensorBatches(columns, _bolt_label_columns, batch_size);

  return std::make_pair(std::move(data), std::move(labels));
}

std::vector<std::pair<uint32_t, RlhfSample>>
MachFeaturizer::getBalancingSamples(
    const dataset::DataSourcePtr& data_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    size_t n_balancing_samples, size_t rows_to_read) {
  auto csv_data_source = dataset::CsvDataSource::make(data_source, _delimiter);

  auto data_iter = data::CsvIterator::make(
      csv_data_source, _delimiter, std::max(n_balancing_samples, rows_to_read));

  auto columns_opt = data_iter->next();
  if (!columns_opt) {
    throw std::invalid_argument("No data found for training.");
  }

  auto columns = std::move(columns_opt.value());

  if (!strong_column_names.empty() || !weak_column_names.empty()) {
    columns = coldStartTransform(strong_column_names, weak_column_names)
                  ->apply(columns, *_state);
  }

  columns = _label_transform->apply(columns, *_state);

  columns.shuffle();

  auto text_col =
      columns.getValueColumn<std::string>(textDatasetConfig().textColumn());
  auto mach_label_col = columns.getArrayColumn<uint32_t>(MACH_LABELS);
  auto doc_id_col = columns.getArrayColumn<uint32_t>(MACH_DOC_IDS);

  size_t n_to_return = std::min(n_balancing_samples, columns.numRows());
  std::vector<std::pair<uint32_t, RlhfSample>> samples;
  for (size_t i = 0; i < n_to_return; i++) {
    auto labels = mach_label_col->row(i);

    RlhfSample sample(text_col->value(i),
                      std::vector<uint32_t>(labels.begin(), labels.end()));

    samples.emplace_back(doc_id_col->row(i)[0], std::move(sample));
  }

  return samples;
}

data::ColumnMap MachFeaturizer::removeIntermediateColumns(
    const data::ColumnMap& columns) {
  std::unordered_map<std::string, data::ColumnPtr> new_columns;
  for (const auto& column : _bolt_input_columns) {
    new_columns[column.indices()] = columns.getColumn(column.indices());
    if (column.values()) {
      new_columns[*column.values()] = columns.getColumn(*column.values());
    }
  }
  for (const auto& column : _bolt_label_columns) {
    new_columns[column.indices()] = columns.getColumn(column.indices());
    if (column.values()) {
      new_columns[*column.values()] = columns.getColumn(*column.values());
    }
  }
  return data::ColumnMap(std::move(new_columns));
}

data::TransformationPtr MachFeaturizer::makeDocIdTransformation(
    const std::string& label_column_name,
    const CategoricalDataTypePtr& label_column_info) {
  if (auto delim = label_column_info->delimiter) {
    return std::make_shared<data::StringToTokenArray>(
        label_column_name, MACH_DOC_IDS, *delim,
        std::numeric_limits<uint32_t>::max());
  }
  return std::make_shared<data::StringToToken>(
      label_column_name, MACH_DOC_IDS, std::numeric_limits<uint32_t>::max());
}

data::TransformationPtr MachFeaturizer::makeLabelTransformations(
    const std::string& label_column_name,
    const CategoricalDataTypePtr& label_column_info) {
  auto doc_id_transform =
      makeDocIdTransformation(label_column_name, label_column_info);

  auto mach_label_transform =
      std::make_shared<data::MachLabel>(MACH_DOC_IDS, MACH_LABELS);

  return data::Pipeline::make({doc_id_transform, mach_label_transform});
}

void MachFeaturizer::addDummyDocIds(data::ColumnMap& columns) {
  auto dummy_doc_ids = data::ValueColumn<uint32_t>::make(
      std::vector<uint32_t>(columns.numRows(), 0),
      std::numeric_limits<uint32_t>::max());

  columns.setColumn(MACH_DOC_IDS, dummy_doc_ids);
}

ar::ConstArchivePtr MachFeaturizer::toArchive() const {
  auto map = Featurizer::toArchiveMap();

  map->set("doc_id_transform", _doc_id_transform->toArchive());
  map->set("prehashed_label_transform",
           _prehashed_labels_transform->toArchive());

  return map;
}

std::shared_ptr<MachFeaturizer> MachFeaturizer::fromArchive(
    const ar::Archive& archive) {
  return std::make_shared<MachFeaturizer>(archive);
}

MachFeaturizer::MachFeaturizer(const ar::Archive& archive)
    : Featurizer(archive),
      _doc_id_transform(
          data::Transformation::fromArchive(*archive.get("doc_id_transform"))),
      _prehashed_labels_transform(data::Transformation::fromArchive(
          *archive.get("prehashed_label_transform"))) {}

template void MachFeaturizer::serialize(cereal::BinaryInputArchive&);
template void MachFeaturizer::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void MachFeaturizer::serialize(Archive& archive) {
  archive(_input_transform, _const_input_transform, _label_transform,
          _bolt_input_columns, _bolt_label_columns, _delimiter, _state,
          _text_dataset, _doc_id_transform, _prehashed_labels_transform);
}

}  // namespace thirdai::automl