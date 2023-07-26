#include "MachDatasetFactory.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <auto_ml/src/featurization/DataTypes.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/CastString.h>
#include <data/src/transformations/ColdStartText.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/StringConcat.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <limits>
#include <stdexcept>

namespace thirdai::automl {

MachDatasetFactory::MachDatasetFactory(
    const data::ColumnDataTypes& input_data_types,
    dataset::mach::MachIndexPtr mach_index, const data::TabularOptions& options)
    : _delimiter(options.delimiter) {
  if (input_data_types.size() != 2) {
    throw std::invalid_argument("Expected a single text and label column.");
  }
  for (const auto& [name, dtype] : input_data_types) {
    if (auto text = data::asText(dtype)) {
      _cold_start_text_column = name;
      _featurized_input_column = name + "_tokenized";
      _input_transformation = std::make_shared<thirdai::data::TextTokenizer>(
          name, _featurized_input_column, text->tokenizer, text->encoder,
          text->lowercase, options.feature_hash_range);
    } else if (auto categorical = data::asCategorical(dtype)) {
      _cold_start_label_column = name;
      _featurized_entity_id_column = name + "_entity_ids";
      _featurized_mach_label_column = name + "_mach_labels";

      if (categorical->delimiter) {
        _entity_id_transformation =
            std::make_shared<thirdai::data::CastStringToTokenArray>(
                name, _featurized_entity_id_column, *categorical->delimiter,
                std::numeric_limits<uint32_t>::max());
      } else {
        _entity_id_transformation =
            std::make_shared<thirdai::data::CastStringToToken>(
                name, _featurized_entity_id_column,
                std::numeric_limits<uint32_t>::max());
      }

      _prehashed_label_transformation =
          std::make_shared<thirdai::data::CastStringToTokenArray>(
              name, _featurized_mach_label_column, ' ',
              mach_index->numBuckets());
    } else {
      throw std::invalid_argument("Unnsupported column datatype for column '" +
                                  name + "'.");
    }
  }

  _mach_label_transformation = std::make_shared<thirdai::data::MachLabel>(
      _featurized_entity_id_column, _featurized_mach_label_column);

  _state = std::make_shared<thirdai::data::State>(std::move(mach_index));
}

thirdai::data::LoaderPtr MachDatasetFactory::getDataLoader(
    const dataset::DataSourcePtr& data_source, bool include_mach_labels,
    dataset::DatasetShuffleConfig shuffle_config, bool verbose) {
  return getDataLoaderHelper(data_source, include_mach_labels, shuffle_config,
                             verbose, nullptr);
}

thirdai::data::LoaderPtr MachDatasetFactory::getColdStartDataLoader(
    const dataset::DataSourcePtr& data_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, bool include_mach_labels,
    bool fast_approximation, dataset::DatasetShuffleConfig shuffle_config,
    bool verbose) {
  auto cold_start_transformation = createColdStartTransformation(
      strong_column_names, weak_column_names, fast_approximation);

  return getDataLoaderHelper(data_source, include_mach_labels, shuffle_config,
                             verbose, cold_start_transformation);
}

thirdai::data::LoaderPtr MachDatasetFactory::getDataLoaderHelper(
    const dataset::DataSourcePtr& data_source, bool include_mach_labels,
    dataset::DatasetShuffleConfig shuffle_config, bool verbose,
    thirdai::data::TransformationPtr cold_start_transformation) {
  auto csv_data_source = dataset::CsvDataSource::make(data_source, _delimiter);

  thirdai::data::ColumnMapIterator data_iter(data_source, _delimiter);

  std::vector<thirdai::data::TransformationPtr> transformations;
  if (cold_start_transformation) {
    transformations.push_back(std::move(cold_start_transformation));
  }
  transformations.push_back(_input_transformation);
  transformations.push_back(_entity_id_transformation);
  if (include_mach_labels) {
    transformations.push_back(_mach_label_transformation);
  }
  auto transformation_list =
      thirdai::data::TransformationList::make({transformations});

  thirdai::data::IndexValueColumnList label_column_outputs;
  if (include_mach_labels) {
    label_column_outputs.emplace_back(_featurized_mach_label_column,
                                      std::nullopt);
  }
  label_column_outputs.emplace_back(_featurized_entity_id_column, std::nullopt);

  return thirdai::data::Loader::make(data_iter, transformation_list, _state,
                                     {{_featurized_input_column, std::nullopt}},
                                     label_column_outputs,
                                     shuffle_config.min_buffer_size, verbose);
}

bolt::nn::tensor::TensorList MachDatasetFactory::featurizeInput(
    const MapInput& sample) {
  auto columns = thirdai::data::ColumnMap::fromMapInput(sample);

  columns = _input_transformation->apply(columns, *_state);

  return thirdai::data::convertToTensorBatch(
      columns, {{_featurized_input_column, std::nullopt}});
}

bolt::nn::tensor::TensorList MachDatasetFactory::featurizeInputBatch(
    const MapInputBatch& samples) {
  auto columns = thirdai::data::ColumnMap::fromMapInputBatch(samples);

  columns = _input_transformation->apply(columns, *_state);

  return thirdai::data::convertToTensorBatch(
      columns, {{_featurized_input_column, std::nullopt}});
}

TensorList MachDatasetFactory::featurizeInputColdStart(
    MapInput sample, const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names) {
  sample[_cold_start_label_column] =
      "";  // Add dummy value for cold start transformation.

  auto columns = thirdai::data::ColumnMap::fromMapInput(sample);

  columns =
      createColdStartTransformation(strong_column_names, weak_column_names)
          ->apply(columns, *_state);

  columns = _input_transformation->apply(columns, *_state);

  return thirdai::data::convertToTensorBatch(
      columns, {{_featurized_input_column, std::nullopt}});
}

std::pair<TensorList, TensorList> MachDatasetFactory::featurizeTrainingBatch(
    const MapInputBatch& samples, bool prehashed) {
  auto columns = thirdai::data::ColumnMap::fromMapInputBatch(samples);

  columns = _input_transformation->apply(columns, *_state);
  columns = _entity_id_transformation->apply(columns, *_state);
  if (prehashed) {
    columns = _prehashed_label_transformation->apply(columns, *_state);
  } else {
    columns = _mach_label_transformation->apply(columns, *_state);
  }

  auto data = thirdai::data::convertToTensorBatch(
      columns, {{_featurized_input_column, std::nullopt}});

  auto labels = thirdai::data::convertToTensorBatch(
      columns, {{_featurized_mach_label_column, std::nullopt},
                {_featurized_entity_id_column, std::nullopt}});

  return std::make_pair(std::move(data), std::move(labels));
}

thirdai::data::TransformationPtr
MachDatasetFactory::createColdStartTransformation(
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    bool fast_approximation) const {
  if (fast_approximation) {
    std::vector<std::string> all_columns = weak_column_names;
    all_columns.insert(all_columns.end(), strong_column_names.begin(),
                       strong_column_names.end());
    return std::make_shared<thirdai::data::StringConcat>(
        all_columns, _cold_start_text_column);
  }

  return std::make_shared<thirdai::data::ColdStartTextAugmentation>(
      /* strong_column_names= */ strong_column_names,
      /* weak_column_names= */ weak_column_names,
      /* label_column_name= */ _cold_start_label_column,
      /* output_column_name= */ _cold_start_text_column);
}

template void MachDatasetFactory::serialize(cereal::BinaryInputArchive&);
template void MachDatasetFactory::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void MachDatasetFactory::serialize(Archive& archive) {
  archive(_input_transformation, _entity_id_transformation,
          _mach_label_transformation, _prehashed_label_transformation,
          _delimiter, _featurized_input_column, _featurized_entity_id_column,
          _featurized_mach_label_column, _cold_start_text_column,
          _cold_start_label_column, _state);
}

}  // namespace thirdai::automl