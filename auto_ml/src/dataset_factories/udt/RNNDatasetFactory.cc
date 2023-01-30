#include "RNNDatasetFactory.h"
#include <cereal/archives/binary.hpp>
#include <auto_ml/src/dataset_factories/udt/CategoricalMetadata.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/FeatureComposer.h>
#include <dataset/src/RecursionWrapper.h>
#include <dataset/src/blocks/Sequence.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>

namespace thirdai::automl::data {

std::string uniqueName(const ColumnDataTypes& data_types,
                       std::string proposed_name) {
  while (data_types.count(proposed_name)) {
    // Use unusual characters '$' to avoid colliding with other column names;
    proposed_name += "$";
  }
  return proposed_name;
}

RNNDatasetFactoryPtr RNNDatasetFactory::make(ColumnDataTypes data_types,
                                             std::string target_column,
                                             uint32_t n_target_classes,
                                             char delimiter,
                                             uint32_t text_pairgram_word_limit,
                                             bool contextual_columns,
                                             uint32_t hash_range) {
  if (!data_types.count(target_column)) {
    throw std::invalid_argument(
        "Target column provided was not found in data_types.");
  }

  // Increment by 1 for EOS token
  n_target_classes++;

  // This column will contain the predicted target sequence so far.
  // The target column will contain the single next item to be predicted.
  auto intermediate_column = uniqueName(data_types, target_column + "_inter");
  data_types[intermediate_column] = data_types.at(target_column);

  // This column will contain the current prediction step.
  // Only used by SequenceTargetBlock. Input vectors will not encode this.
  auto step_column = uniqueName(data_types, target_column + "_step");

  auto target_sequence = asSequence(data_types.at(intermediate_column));
  auto target_sequence_delimiter = target_sequence->delimiter;
  auto max_recursion_depth = target_sequence->max_length.value();

  CategoricalMetadata metadata(data_types, text_pairgram_word_limit,
                               contextual_columns, hash_range);

  auto input_blocks = FeatureComposer::makeNonTemporalFeatureBlocks(
      /* data_types= */ data_types, /* target= */ target_column,
      /* temporal_relationships= */ {},
      /* vectors_map= */ metadata.metadataVectors(),
      /* text_pairgrams_word_limit= */ text_pairgram_word_limit,
      /* contextual_columns= */ contextual_columns);

  auto label_block = dataset::SequenceTargetBlock::make(
      /* target_col= */ target_column, /* step_col= */ step_column,
      /* max_steps= */ max_recursion_depth,
      /* vocabulary_size= */ n_target_classes);

  auto unlabeled_featurizer = dataset::TabularFeaturizer::make(
      /* input_blocks= */ input_blocks, /* label_blocks= */ {},
      /* has_header= */ true, /* delimiter= */ delimiter, /* parallel= */ true,
      /* hash_range= */ hash_range);

  auto labeled_featurizer = dataset::TabularFeaturizer::make(
      /* input_blocks= */ input_blocks, /* label_blocks= */ {label_block},
      /* has_header= */ true, /* delimiter= */ delimiter, /* parallel= */ true,
      /* hash_range= */ hash_range);

  return std::shared_ptr<RNNDatasetFactory>(new RNNDatasetFactory(
      /* augmented_data_types= */ std::move(data_types),
      /* intermediate_column= */ std::move(intermediate_column),
      /* current_step_target_column= */ std::move(target_column),
      /* step_column= */ step_column, /* delimiter= */ delimiter,
      /* target_sequence_delimiter= */ target_sequence_delimiter,
      /* max_recursion_depth= */ max_recursion_depth,
      /* label_block= */ std::move(label_block),
      /* categorical_metadata= */ std::move(metadata),
      /* unlabeled_featurizer= */ std::move(unlabeled_featurizer),
      /* labeled_featurizer= */ std::move(labeled_featurizer)));
}

dataset::DatasetLoaderPtr RNNDatasetFactory::getLabeledDatasetLoader(
    dataset::DataSourcePtr source, bool training) {
  auto data_source = dataset::RecursionWrapper::make(
      /* source= */ std::move(source),
      /* column_delimiter= */ _delimiter,
      /* target_delimiter= */ _target_delimiter,
      /* intermediate_column= */ _intermediate_column,
      /* target_column= */ _current_step_target_column,
      /* step_column= */ _step_column,
      /* max_recursion_depth= */ _max_recursion_depth);

  return std::make_unique<dataset::DatasetLoader>(
      data_source, _labeled_featurizer, /* shuffle= */ training);
}

uint32_t RNNDatasetFactory::labelToNeuronId(
    std::variant<uint32_t, std::string> label) {
  (void)label;
  throw std::invalid_argument(
      "Recursive model currently does not support explanation.");
}

std::string RNNDatasetFactory::className(uint32_t neuron_id) const {
  return _label_block->className(neuron_id);
}

std::string RNNDatasetFactory::classNameAtStep(const BoltVector& activations,
                                               uint32_t step) const {
  return _label_block->classNameAtStep(activations, step);
}

void RNNDatasetFactory::incorporateNewPrediction(
    MapInput& sample, const std::string& new_prediction) const {
  auto& intermediate_column = sample[_intermediate_column];
  if (!intermediate_column.empty()) {
    intermediate_column += _target_delimiter;
  }
  intermediate_column += new_prediction;
}

std::string RNNDatasetFactory::stitchTargetSequence(
    const std::vector<std::string>& predictions) const {
  if (predictions.empty()) {
    return "";
  }
  std::stringstream output_sequence;
  output_sequence << predictions.front();
  for (uint32_t i = 1; i < predictions.size(); i++) {
    output_sequence << _target_delimiter << predictions[i];
  }
  return output_sequence.str();
}

void RNNDatasetFactory::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void RNNDatasetFactory::save_stream(std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

RNNDatasetFactoryPtr RNNDatasetFactory::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

RNNDatasetFactoryPtr RNNDatasetFactory::load_stream(
    std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<RNNDatasetFactory> deserialize_into(new RNNDatasetFactory());
  iarchive(*deserialize_into);
  return deserialize_into;
}

// This private constructor is only called by make()
RNNDatasetFactory::RNNDatasetFactory(
    ColumnDataTypes augmented_data_types, std::string intermediate_column,
    std::string current_step_target_column, std::string step_column,
    char delimiter, char target_sequence_delimiter,
    uint32_t max_recursion_depth, dataset::SequenceTargetBlockPtr label_block,
    CategoricalMetadata categorical_metadata,
    dataset::TabularFeaturizerPtr unlabeled_featurizer,
    dataset::TabularFeaturizerPtr labeled_featurizer)
    : _data_types(std::move(augmented_data_types)),
      _intermediate_column(std::move(intermediate_column)),
      _current_step_target_column(std::move(current_step_target_column)),
      _step_column(std::move(step_column)),
      _delimiter(delimiter),
      _target_delimiter(target_sequence_delimiter),
      _max_recursion_depth(max_recursion_depth),
      _label_block(std::move(label_block)),
      _categorical_metadata(std::move(categorical_metadata)),
      _unlabeled_featurizer(std::move(unlabeled_featurizer)),
      _labeled_featurizer(std::move(labeled_featurizer)) {}

}  // namespace thirdai::automl::data

CEREAL_REGISTER_TYPE(thirdai::automl::data::RNNDatasetFactory)