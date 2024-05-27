#include "TabularDatasetFactory.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/unordered_map.hpp>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/featurization/TabularBlockComposer.h>
#include <auto_ml/src/udt/Defaults.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/BlockList.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace thirdai::automl {

TabularDatasetFactory::TabularDatasetFactory(
    ColumnDataTypes data_types,
    const UserProvidedTemporalRelationships& provided_temporal_relationships,
    const std::vector<dataset::BlockList>& label_blocks,
    std::set<std::string> label_col_names, const TabularOptions& options,
    bool force_parallel)
    : _data_types(std::move(data_types)),
      _label_col_names(std::move(label_col_names)),
      _num_label_blocks(label_blocks.size()),
      _options(options) {
  _vectors_map = processAllMetadata(_data_types, options);

  TemporalRelationships temporal_relationships =
      TemporalRelationshipsAutotuner::autotune(
          _data_types, provided_temporal_relationships, options.lookahead);

  bool parallel = force_parallel || temporal_relationships.empty();

  _labeled_featurizer = makeFeaturizer(temporal_relationships,
                                       /* should_update_history= */ true,
                                       options, label_blocks, parallel);
  _inference_featurizer =
      makeFeaturizer(temporal_relationships,
                     /* should_update_history= */ false, options,
                     /* label_blocks = */ {}, parallel);
}

dataset::DatasetLoaderPtr TabularDatasetFactory::getLabeledDatasetLoader(
    const dataset::DataSourcePtr& data_source, bool shuffle,
    std::optional<dataset::DatasetShuffleConfig> shuffle_config) {
  if (!shuffle_config.has_value()) {
    shuffle_config = dataset::DatasetShuffleConfig();
  }

  auto csv_data_source = dataset::CsvDataSource::make(data_source, delimiter());

  return std::make_unique<dataset::DatasetLoader>(
      csv_data_source, _labeled_featurizer,
      /* shuffle= */ shuffle, shuffle_config.value());
}

dataset::DatasetLoaderPtr TabularDatasetFactory::getUnLabeledDatasetLoader(
    const dataset::DataSourcePtr& data_source) {
  auto csv_data_source = dataset::CsvDataSource::make(data_source, delimiter());
  return std::make_unique<dataset::DatasetLoader>(
      csv_data_source, _inference_featurizer, /* shuffle= */ false);
}

bolt::TensorList TabularDatasetFactory::featurizeInputBatch(
    const MapInputBatch& inputs) {
  verifyValidColNames(inputs);

  dataset::MapBatchRef inputs_ref(inputs);

  std::vector<BoltBatch> result;

  result.emplace_back(
      std::move(_inference_featurizer->featurize(inputs_ref).at(0)));

  return bolt::convertBatch(std::move(result),
                            _inference_featurizer->getDimensions());
}

std::pair<bolt::TensorList, bolt::TensorList>
TabularDatasetFactory::featurizeTrainingBatch(const MapInputBatch& batch) {
  verifyValidColNames(batch);

  dataset::MapBatchRef inputs_ref(batch);

  auto featurized = _labeled_featurizer->featurize(inputs_ref);
  auto dims = _labeled_featurizer->getDimensions();

  uint32_t num_data_blocks = dims.size() - _num_label_blocks;

  bolt::TensorList data;
  for (uint32_t i = 0; i < num_data_blocks; i++) {
    data.push_back(
        bolt::Tensor::convert(BoltBatch(std::move(featurized[i])), dims[i]));
  }

  bolt::TensorList labels;
  for (uint32_t i = num_data_blocks; i < num_data_blocks + _num_label_blocks;
       i++) {
    labels.push_back(
        bolt::Tensor::convert(BoltBatch(std::move(featurized[i])), dims[i]));
  }

  return {std::move(data), std::move(labels)};
}

void TabularDatasetFactory::updateMetadata(const std::string& col_name,
                                           const MapInput& update) {
  verifyColumnMetadataExists(col_name);

  auto metadata_config = getColumnMetadataConfig(col_name);

  dataset::MapSampleRef update_ref(update);
  auto vec = _metadata_processors.at(col_name)->featurize(update_ref).at(0);

  const auto& key = update.at(metadata_config->key);
  _vectors_map.at(col_name)->vectors[key] = vec;
}

void TabularDatasetFactory::updateMetadataBatch(const std::string& col_name,
                                                const MapInputBatch& updates) {
  verifyColumnMetadataExists(col_name);
  auto metadata_config = getColumnMetadataConfig(col_name);

  dataset::MapBatchRef updates_ref(updates);
  std::vector<BoltVector> batch =
      _metadata_processors.at(col_name)->featurize(updates_ref).at(0);

  for (uint32_t update_idx = 0; update_idx < updates.size(); update_idx++) {
    const auto& key = updates.at(update_idx).at(metadata_config->key);
    _vectors_map.at(col_name)->vectors[key] = batch.at(update_idx);
  }
}

dataset::TabularFeaturizerPtr TabularDatasetFactory::makeFeaturizer(
    const TemporalRelationships& temporal_relationships,
    bool should_update_history, const TabularOptions& options,
    const std::vector<dataset::BlockList>& label_blocks, bool parallel) {
  auto input_blocks = makeTabularInputBlocks(
      _data_types, _label_col_names, temporal_relationships, _vectors_map,
      _temporal_context, should_update_history, options);

  std::vector<dataset::BlockList> block_lists = {
      dataset::BlockList(std::move(input_blocks),
                         /* hash_range= */ options.feature_hash_range)};

  block_lists.insert(block_lists.end(), label_blocks.begin(),
                     label_blocks.end());

  return dataset::TabularFeaturizer::make(
      /* block_lists = */ block_lists,
      /* has_header= */ true,
      /* delimiter= */ options.delimiter, /* parallel= */ parallel);
}

PreprocessedVectorsMap TabularDatasetFactory::processAllMetadata(
    const ColumnDataTypes& input_data_types, const TabularOptions& options) {
  PreprocessedVectorsMap metadata_vectors;
  for (const auto& [col_name, col_type] : input_data_types) {
    if (auto categorical = asCategorical(col_type)) {
      if (categorical->metadata_config) {
        metadata_vectors[col_name] = makeProcessedVectorsForCategoricalColumn(
            col_name, categorical, options);
      }
    }
  }
  return metadata_vectors;
}

dataset::PreprocessedVectorsPtr
TabularDatasetFactory::makeProcessedVectorsForCategoricalColumn(
    const std::string& col_name, const CategoricalDataTypePtr& categorical,
    const TabularOptions& options) {
  if (!categorical->metadata_config) {
    throw std::invalid_argument("The given categorical column (" + col_name +
                                ") does not have a metadata config.");
  }

  auto metadata = categorical->metadata_config;

  auto data_source = dataset::FileDataSource::make(metadata->metadata_file);

  auto input_blocks = makeNonTemporalInputBlocks(
      metadata->column_data_types, {metadata->key}, {}, {}, options);

  auto key_vocab = dataset::ThreadSafeVocabulary::make();
  auto label_block =
      dataset::StringLookupCategoricalBlock::make(metadata->key, key_vocab);

  _metadata_processors[col_name] = dataset::TabularFeaturizer::make(
      /* block_lists = */ {dataset::BlockList(
                               std::move(input_blocks),
                               /* hash_range= */ options.feature_hash_range),
                           dataset::BlockList({label_block})},
      /* has_header= */ true,
      /* delimiter= */ metadata->delimiter);

  // Here we set parallel=true because there are no temporal
  // relationships in the metadata file.
  dataset::DatasetLoader metadata_source(
      /* data_source= */ data_source,
      /* featurizer= */ _metadata_processors[col_name],
      /* shuffle = */ false);

  return preprocessedVectorsFromDataset(metadata_source, *key_vocab);
}

dataset::PreprocessedVectorsPtr
TabularDatasetFactory::preprocessedVectorsFromDataset(
    dataset::DatasetLoader& dataset_loader,
    dataset::ThreadSafeVocabulary& key_vocab) {
  // The batch size does not really matter here because we are storing these
  // vectors as metadata, not training on them. Thus, we choose the somewhat
  // arbitrary value 2048 since it is large enough to use all threads.
  auto datasets =
      dataset_loader.loadAll(/* batch_size = */ udt::defaults::BATCH_SIZE);

  if (datasets.size() != 2) {
    throw std::runtime_error(
        "The featurizer for preprocessed vectors should return only a single "
        "input and label");
  }
  auto vectors = datasets.at(0);
  auto ids = datasets.at(1);

  std::unordered_map<std::string, BoltVector> preprocessed_vectors(ids->len());

  for (uint32_t batch = 0; batch < vectors->numBatches(); batch++) {
    for (uint32_t vec = 0; vec < vectors->at(batch).getBatchSize(); vec++) {
      auto id = ids->at(batch)[vec].active_neurons[0];
      auto key = key_vocab.getString(id);
      preprocessed_vectors[key] = std::move(vectors->at(batch)[vec]);
    }
  }

  return std::make_shared<dataset::PreprocessedVectors>(
      std::move(preprocessed_vectors), dataset_loader.getInputDim());
}

void TabularDatasetFactory::save_stream(std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

std::shared_ptr<TabularDatasetFactory> TabularDatasetFactory::load_stream(
    std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<TabularDatasetFactory> deserialize_into(
      new TabularDatasetFactory());
  iarchive(*deserialize_into);

  return deserialize_into;
}

template void TabularDatasetFactory::serialize(cereal::BinaryInputArchive&);
template void TabularDatasetFactory::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void TabularDatasetFactory::serialize(Archive& archive) {
  archive(_labeled_featurizer, _inference_featurizer, _metadata_processors,
          _vectors_map, _temporal_context, _data_types, _label_col_names,
          _options);
}
}  // namespace thirdai::automl