#include "MachDatasetFactory.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <auto_ml/src/featurization/DataTypes.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/ColdStartText.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringConcat.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <utils/UUID.h>
#include <limits>
#include <stdexcept>

namespace thirdai::automl {

MachDatasetFactory::MachDatasetFactory(data::ColumnDataTypes input_data_types,
                                       dataset::mach::MachIndexPtr mach_index,
                                       const data::TabularOptions& options)
    : _input_data_types(std::move(input_data_types)),
      _delimiter(options.delimiter) {
  _featurized_input_column_name = uniqueColumnName("__featurized_input__");
  _entity_id_column_name = uniqueColumnName("__entity_ids__");
  _mach_label_column_name = uniqueColumnName("__mach_labels__");

  if (_input_data_types.size() != 2) {
    throw std::invalid_argument("Expected a single text and label column.");
  }
  for (const auto& [name, dtype] : _input_data_types) {
    if (auto text = data::asText(dtype)) {
      _input_text_column_name = name;
      _input_featurization = std::make_shared<thirdai::data::TextTokenizer>(
          /* input_column= */ name,
          /* output_column= */ _featurized_input_column_name,
          /* tokenizer= */ text->tokenizer,
          /* encoder= */ text->encoder,
          /* lower_case= */ text->lowercase,
          /* dim= */ options.feature_hash_range);
    } else if (auto categorical = data::asCategorical(dtype)) {
      _input_label_column_name = name;

      if (categorical->delimiter) {
        _strings_to_entity_ids =
            std::make_shared<thirdai::data::StringToTokenArray>(
                /* input_column_name= */ name,
                /* output_column_name= */ _entity_id_column_name,
                /* delimiter= */ *categorical->delimiter,
                /* dim= */ std::numeric_limits<uint32_t>::max());
      } else {
        _strings_to_entity_ids = std::make_shared<thirdai::data::StringToToken>(
            /* input_column_name= */ name,
            /* output_column_name= */ _entity_id_column_name,
            /* dim= */ std::numeric_limits<uint32_t>::max());
      }

      _strings_to_prehashed_mach_labels =
          std::make_shared<thirdai::data::StringToTokenArray>(
              /* input_column_name= */ name,
              /* output_column_name= */ _mach_label_column_name,
              /* delimiter= */ ' ',
              /* dim= */ mach_index->numBuckets());
    } else {
      throw std::invalid_argument("Unnsupported column datatype for column '" +
                                  name + "'.");
    }
  }

  _entity_ids_to_mach_labels = std::make_shared<thirdai::data::MachLabel>(
      /* input_column_name= */ _entity_id_column_name,
      /* output_column_name= */ _mach_label_column_name);

  _state = std::make_shared<thirdai::data::State>(std::move(mach_index));
}

thirdai::data::LoaderPtr MachDatasetFactory::getDataLoader(
    const dataset::DataSourcePtr& data_source, bool include_mach_labels,
    size_t batch_size, bool shuffle, bool verbose,
    dataset::DatasetShuffleConfig shuffle_config) {
  return getDataLoaderHelper(data_source,
                             /* include_mach_labels= */ include_mach_labels,
                             /* batch_size= */ batch_size,
                             /* shuffle= */ shuffle,
                             /* verbose= */ verbose,
                             /* shuffle_config= */ shuffle_config,
                             /* cold_start_transformation= */ nullptr);
}

thirdai::data::LoaderPtr MachDatasetFactory::getColdStartDataLoader(
    const dataset::DataSourcePtr& data_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, bool fast_approximation,
    bool include_mach_labels, size_t batch_size, bool shuffle, bool verbose,
    dataset::DatasetShuffleConfig shuffle_config) {
  auto cold_start_transformation = createColdStartTransformation(
      strong_column_names, weak_column_names, fast_approximation);

  return getDataLoaderHelper(
      data_source,
      /* include_mach_labels= */ include_mach_labels,
      /* batch_size= */ batch_size, /* shuffle= */ shuffle,
      /* verbose= */ verbose,
      /* shuffle_config= */ shuffle_config,
      /* cold_start_transformation= */ cold_start_transformation);
}

thirdai::data::LoaderPtr MachDatasetFactory::getDataLoaderHelper(
    const dataset::DataSourcePtr& data_source, bool include_mach_labels,
    size_t batch_size, bool shuffle, bool verbose,
    dataset::DatasetShuffleConfig shuffle_config,
    thirdai::data::TransformationPtr cold_start_transformation) {
  auto csv_data_source = dataset::CsvDataSource::make(data_source, _delimiter);

  thirdai::data::ColumnMapIterator data_iter(data_source, _delimiter);

  std::vector<thirdai::data::TransformationPtr> transformations;
  if (cold_start_transformation) {
    transformations.push_back(std::move(cold_start_transformation));
  }
  transformations.push_back(_input_featurization);
  transformations.push_back(_strings_to_entity_ids);
  if (include_mach_labels) {
    transformations.push_back(_entity_ids_to_mach_labels);
  }
  auto transformation_list =
      thirdai::data::TransformationList::make(transformations);

  thirdai::data::IndexValueColumnList label_column_outputs;
  if (include_mach_labels) {
    label_column_outputs.emplace_back(_mach_label_column_name, std::nullopt);
  }
  label_column_outputs.emplace_back(_entity_id_column_name, std::nullopt);

  return thirdai::data::Loader::make(
      data_iter, transformation_list, _state,
      {{_featurized_input_column_name, std::nullopt}}, label_column_outputs,
      /* batch_size= */ batch_size, /* shuffle= */ shuffle,
      /* verbose= */ verbose,
      /* shuffle_buffer_size= */ shuffle_config.min_buffer_size,
      /* shuffle_seed= */ shuffle_config.seed);
}

bolt::nn::tensor::TensorList MachDatasetFactory::featurizeInput(
    const MapInput& sample) {
  auto columns = thirdai::data::ColumnMap::fromMapInput(sample);

  columns = _input_featurization->apply(columns, *_state);

  return thirdai::data::toTensors(
      columns, {{_featurized_input_column_name, std::nullopt}});
}

bolt::nn::tensor::TensorList MachDatasetFactory::featurizeInputBatch(
    const MapInputBatch& samples) {
  auto columns = thirdai::data::ColumnMap::fromMapInputBatch(samples);

  columns = _input_featurization->apply(columns, *_state);

  return thirdai::data::toTensors(
      columns, {{_featurized_input_column_name, std::nullopt}});
}

TensorList MachDatasetFactory::featurizeInputColdStart(
    MapInput sample, const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names) {
  // Currently the cold start transformation expects a label column that it
  // repeats when it maps a single set of phrases to multiple samples. In the
  // future it should be extended to not require this, and just duplicate any
  // other columns it finds.
  sample[_input_label_column_name] = "";

  auto columns = thirdai::data::ColumnMap::fromMapInput(sample);

  columns =
      createColdStartTransformation(strong_column_names, weak_column_names)
          ->apply(columns, *_state);

  columns = _input_featurization->apply(columns, *_state);

  return thirdai::data::toTensors(
      columns, {{_featurized_input_column_name, std::nullopt}});
}

std::pair<TensorList, TensorList> MachDatasetFactory::featurizeTrainingBatch(
    const MapInputBatch& samples, bool prehashed) {
  auto columns = thirdai::data::ColumnMap::fromMapInputBatch(samples);

  columns = _input_featurization->apply(columns, *_state);
  columns = _strings_to_entity_ids->apply(columns, *_state);
  if (prehashed) {
    columns = _strings_to_prehashed_mach_labels->apply(columns, *_state);
  } else {
    columns = _entity_ids_to_mach_labels->apply(columns, *_state);
  }

  auto data = thirdai::data::toTensors(
      columns, {{_featurized_input_column_name, std::nullopt}});

  auto labels = thirdai::data::toTensors(
      columns, {{_mach_label_column_name, std::nullopt},
                {_entity_id_column_name, std::nullopt}});

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
        all_columns, _input_text_column_name);
  }

  return std::make_shared<thirdai::data::ColdStartTextAugmentation>(
      /* strong_column_names= */ strong_column_names,
      /* weak_column_names= */ weak_column_names,
      /* label_column_name= */ _input_label_column_name,
      /* output_column_name= */ _input_text_column_name);
}

std::string MachDatasetFactory::uniqueColumnName(std::string name) const {
  while (_input_data_types.count(name)) {
    name += utils::uuid::getRandomHexString(/* num_bytes_randomness= */ 4);
  }

  return name;
}

template void MachDatasetFactory::serialize(cereal::BinaryInputArchive&);
template void MachDatasetFactory::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void MachDatasetFactory::serialize(Archive& archive) {
  archive(_input_featurization, _strings_to_entity_ids,
          _entity_ids_to_mach_labels, _strings_to_prehashed_mach_labels,
          _delimiter, _featurized_input_column_name, _entity_id_column_name,
          _mach_label_column_name, _input_text_column_name,
          _input_label_column_name, _state);
}

}  // namespace thirdai::automl