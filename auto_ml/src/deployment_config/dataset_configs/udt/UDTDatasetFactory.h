#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include "ColumnNumberMap.h"
#include "DataTypes.h"
#include "FeatureComposer.h"
#include "TemporalContext.h"
#include "TemporalRelationshipsAutotuner.h"
#include "UDTConfig.h"
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <dataset/src/DataLoader.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/utils/PreprocessedVectors.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace thirdai::automl::deployment {

class UDTDatasetFactory final : public DatasetLoaderFactory {
 public:
  explicit UDTDatasetFactory(UDTConfigPtr config, bool force_parallel,
                             uint32_t text_pairgram_word_limit,
                             bool contextual_columns = false)
      : _config(std::move(config)),
        _temporal_relationships(TemporalRelationshipsAutotuner::autotune(
            _config->data_types, _config->provided_relationships,
            _config->lookahead)),
        _context(std::make_shared<TemporalContext>()),
        _parallel(_temporal_relationships.empty() || force_parallel),
        _text_pairgram_word_limit(text_pairgram_word_limit),
        _contextual_columns(contextual_columns) {
    FeatureComposer::verifyConfigIsValid(*_config, _temporal_relationships);

    _vectors_map = processAllMetadata();

    ColumnNumberMap mock_column_number_map(_config->data_types);
    auto mock_processor = makeLabeledUpdatingProcessor(mock_column_number_map);

    _input_dim = mock_processor->getInputDim();
    _label_dim = mock_processor->getLabelDim();
  }

  static std::shared_ptr<UDTDatasetFactory> make(
      UDTConfigPtr config, bool force_parallel,
      uint32_t text_pairgram_word_limit, bool contextual_columns = false) {
    return std::make_shared<UDTDatasetFactory>(
        std::move(config), force_parallel, text_pairgram_word_limit,
        contextual_columns);
  }

  DatasetLoaderPtr getLabeledDatasetLoader(
      std::shared_ptr<dataset::DataLoader> data_loader, bool training) final {
    auto current_column_number_map =
        makeColumnNumberMap(*data_loader, _config->delimiter);

    if (!_column_number_map) {
      _column_number_map = std::move(current_column_number_map);
      _column_number_to_name = _column_number_map->getColumnNumToColNameMap();
    } else if (!_column_number_map->equals(*current_column_number_map)) {
      throw std::invalid_argument("Column positions should not change.");
    }

    if (!_labeled_history_updating_processor) {
      _labeled_history_updating_processor =
          makeLabeledUpdatingProcessor(*_column_number_map);
    }

    // We initialize the inference batch processor here because we need the
    // column number map.
    if (!_unlabeled_non_updating_processor) {
      _unlabeled_non_updating_processor =
          makeUnlabeledNonUpdatingProcessor(*_column_number_map);
    }

    // The batch processor will treat the next line as a header
    // Restart so batch processor does not skip a sample.
    data_loader->restart();

    return std::make_unique<GenericDatasetLoader>(
        data_loader, _labeled_history_updating_processor,
        /* shuffle= */ training);
  }

  std::vector<BoltVector> featurizeInput(const LineInput& input) final {
    return featurizeInputImpl(input, /* should_update_history= */ false);
  }

  std::vector<BoltVector> featurizeInput(const MapInput& input) final {
    return featurizeInputImpl(input, /* should_update_history= */ false);
  }

  std::vector<BoltVector> updateTemporalTrackers(const LineInput& input) {
    return featurizeInputImpl(input, /* should_update_history= */ true);
  }

  std::vector<BoltVector> updateTemporalTrackers(const MapInput& input) {
    return featurizeInputImpl(input, /* should_update_history= */ true);
  }

  std::vector<BoltBatch> featurizeInputBatch(
      const LineInputBatch& inputs) final {
    return featurizeInputBatchImpl(inputs, /* should_update_history= */ false);
  }

  std::vector<BoltBatch> featurizeInputBatch(
      const MapInputBatch& inputs) final {
    return featurizeInputBatchImpl(lineInputBatchFromMapInputBatch(inputs),
                                   /* should_update_history= */ false);
  }

  uint32_t labelToNeuronId(std::variant<uint32_t, std::string> label) final {
    if (std::holds_alternative<uint32_t>(label)) {
      if (!_config->data_types.at(_config->target)
               .asCategorical()
               .contiguous_numerical_ids) {
        throw std::invalid_argument(
            "Received an integer label but the target column does not contain "
            "contiguous numerical IDs (the contiguous_numerical_id option of "
            "the categorical data type is set to false). Label must be passed "
            "in as a string.");
      }
      return std::get<uint32_t>(label);
    }

    const std::string& label_str = std::get<std::string>(label);

    if (_config->data_types.at(_config->target)
            .asCategorical()
            .contiguous_numerical_ids) {
      throw std::invalid_argument(
          "Received a string label but the target column contains contiguous "
          "numerical IDs (the contiguous_numerical_id option of the "
          "categorical data type is set to true). Label must be passed in as "
          "an integer.");
    }

    if (!_vocabs.count(_config->target)) {
      throw std::invalid_argument(
          "Attempted to get label to neuron id map before training.");
    }
    return _vocabs.at(_config->target)->getUid(label_str);
  }

  std::string className(uint32_t neuron_id) const final {
    if (!_vocabs.count(_config->target)) {
      throw std::invalid_argument(
          "Attempted to get id to label map before training.");
    }
    if (_config->data_types.at(_config->target)
            .asCategorical()
            .contiguous_numerical_ids) {
      throw std::invalid_argument(
          "This model does not provide a mapping from ids to labels since the "
          "target column has contiguous numerical ids; the ids and labels are "
          "equivalent.");
    }
    return _vocabs.at(_config->target)->getString(neuron_id);
  }

  std::vector<BoltBatch> batchUpdateTemporalTrackers(
      const LineInputBatch& inputs) {
    return featurizeInputBatchImpl(inputs, /* should_update_history= */ true);
  }

  std::vector<BoltBatch> batchUpdateTemporalTrackers(
      const MapInputBatch& inputs) {
    return featurizeInputBatchImpl(lineInputBatchFromMapInputBatch(inputs),
                                   /* should_update_history= */ true);
  }

  void resetTemporalTrackers() { _context->reset(); }

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio,
      const LineInput& sample) final {
    return explainImpl(gradients_indices, gradients_ratio, sample);
  }

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio, const MapInput& sample) final {
    return explainImpl(gradients_indices, gradients_ratio, sample);
  }

  std::vector<bolt::InputPtr> getInputNodes() final {
    return {bolt::Input::make(_input_dim)};
  }

  uint32_t getLabelDim() final { return _label_dim; }

 private:
  PreprocessedVectorsMap processAllMetadata() {
    PreprocessedVectorsMap metadata_vectors;
    for (const auto& [col_name, col_type] : _config->data_types) {
      if (col_type.isCategorical()) {
        auto categorical = col_type.asCategorical();
        if (categorical.metadata_config) {
          metadata_vectors[col_name] =
              makeProcessedVectorsForCategoricalColumn(col_name, categorical);
        }
      }
    }
    return metadata_vectors;
  }

  dataset::PreprocessedVectorsPtr makeProcessedVectorsForCategoricalColumn(
      const std::string& col_name, const CategoricalDataType& categorical) {
    if (!categorical.metadata_config) {
      throw std::invalid_argument("The given categorical column (" + col_name +
                                  ") does not have a metadata config.");
    }

    auto metadata = categorical.metadata_config;

    auto data_loader =
        dataset::SimpleFileDataLoader::make(metadata->metadata_file,
                                            /* target_batch_size= */ 2048);

    auto column_numbers =
        makeColumnNumberMap(*data_loader, metadata->delimiter);

    auto input_blocks = buildMetadataInputBlocks(*metadata, *column_numbers);

    auto key_vocab = dataset::ThreadSafeVocabulary::make(
        /* vocab_size= */ 0, /* limit_vocab_size= */ false);
    auto label_block = dataset::StringLookupCategoricalBlock::make(
        column_numbers->at(metadata->key), key_vocab);

    dataset::StreamingGenericDatasetLoader metadata_loader(
        /* loader= */ data_loader,
        /* processor= */
        dataset::GenericBatchProcessor::make(
            /* input_blocks= */ std::move(input_blocks),
            /* label_blocks= */ {std::move(label_block)},
            /* has_header= */ false, /* delimiter= */ metadata->delimiter,
            /* parallel= */ true, /* hash_range= */ _config->hash_range));

    return preprocessedVectorsFromDataset(metadata_loader, *key_vocab);
  }

  static ColumnNumberMapPtr makeColumnNumberMap(
      dataset::DataLoader& data_loader, char delimiter) {
    auto header = data_loader.nextLine();
    if (!header) {
      throw std::invalid_argument(
          "The dataset must have a header that contains column names.");
    }

    return std::make_shared<ColumnNumberMap>(*header, delimiter);
  }

  std::vector<dataset::BlockPtr> buildMetadataInputBlocks(
      const CategoricalMetadataConfig& metadata_config,
      const ColumnNumberMap& column_numbers) const {
    UDTConfig feature_config(
        /* data_types= */ metadata_config.column_data_types,
        /* temporal_tracking_relationships= */ {},
        /* target= */ metadata_config.key,
        /* n_target_classes= */ 0);
    TemporalRelationships empty_temporal_relationships;

    PreprocessedVectorsMap empty_vectors_map;

    return FeatureComposer::makeNonTemporalFeatureBlocks(
        feature_config, empty_temporal_relationships, column_numbers,
        empty_vectors_map, _text_pairgram_word_limit, _contextual_columns);
  }

  static dataset::PreprocessedVectorsPtr preprocessedVectorsFromDataset(
      dataset::StreamingGenericDatasetLoader& dataset,
      dataset::ThreadSafeVocabulary& key_vocab) {
    auto [vectors, ids] = dataset.loadInMemory();

    std::unordered_map<std::string, BoltVector> preprocessed_vectors(
        vectors->len());

    for (uint32_t batch = 0; batch < vectors->numBatches(); batch++) {
      for (uint32_t vec = 0; vec < vectors->at(batch).getBatchSize(); vec++) {
        auto id = ids->at(batch)[vec].active_neurons[0];
        auto key = key_vocab.getString(id);
        preprocessed_vectors[key] = std::move(vectors->at(batch)[vec]);
      }
    }

    return std::make_shared<dataset::PreprocessedVectors>(
        std::move(preprocessed_vectors), dataset.getInputDim());
  }

  template <typename InputType>
  std::vector<BoltVector> featurizeInputImpl(const InputType& input,
                                             bool should_update_history) {
    verifyProcessorsAreInitialized();

    BoltVector vector;
    auto sample = toVectorOfStringViews(input);
    if (auto exception = getProcessor(should_update_history)
                             .makeInputVector(sample, vector)) {
      std::rethrow_exception(exception);
    }
    return {std::move(vector)};
  }

  std::vector<BoltBatch> featurizeInputBatchImpl(const LineInputBatch& inputs,
                                                 bool should_update_history) {
    verifyProcessorsAreInitialized();
    auto [input_batch, _] =
        getProcessor(should_update_history).createBatch(inputs);

    // We cannot use the initializer list because the copy constructor is
    // deleted for BoltBatch.
    std::vector<BoltBatch> batch_list;
    batch_list.emplace_back(std::move(input_batch));
    return batch_list;
  }

  template <typename InputType>
  std::vector<dataset::Explanation> explainImpl(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio, const InputType& sample) {
    verifyProcessorsAreInitialized();

    auto input_row = toVectorOfStringViews(sample);
    auto result = bolt::getSignificanceSortedExplanations(
        gradients_indices, gradients_ratio, input_row,
        _unlabeled_non_updating_processor);

    for (auto& response : result) {
      // We need this conditional because tabular pairgram block provides its
      // own column name.
      if (response.column_name.empty()) {
        response.column_name = _column_number_to_name[response.column_number];
      }
    }

    return result;
  }

  /**
   * The labeled updating processor is used for training and evaluation, which
   * automatically updates the temporal context, as well as for manually
   * updating the temporal context.
   */
  dataset::GenericBatchProcessorPtr makeLabeledUpdatingProcessor(
      const ColumnNumberMap& column_number_map) {
    if (!_config->data_types.count(_config->target)) {
      throw std::invalid_argument(
          "data_types parameter must include the target column.");
    }

    auto target_type = _config->data_types.at(_config->target);
    if (!target_type.isCategorical()) {
      throw std::invalid_argument(
          "Target column must be a categorical column.");
    }

    auto col_num = column_number_map.at(_config->target);
    auto target_config = target_type.asCategorical();
    
    dataset::BlockPtr label_block;
    if (target_config.contiguous_numerical_ids) {
      label_block = dataset::NumericalCategoricalBlock::make(
          /* col= */ col_num, /* n_classes= */ _config->n_target_classes,
          /* delimiter= */ target_config.delimiter);
    } else {
      if (!_vocabs.count(_config->target)) {
        _vocabs[_config->target] = dataset::ThreadSafeVocabulary::make(
            /* vocab_size= */ _config->n_target_classes);
      }
      label_block = dataset::StringLookupCategoricalBlock::make(
          /* col= */ col_num, /* vocab= */ _vocabs.at(_config->target),
          /* delimiter= */ target_config.delimiter);
    }

    auto input_blocks =
        buildInputBlocks(/* column_numbers= */ column_number_map,
                         /* should_update_history= */ true);

    auto processor = dataset::GenericBatchProcessor::make(
        std::move(input_blocks), {label_block}, /* has_header= */ true,
        /* delimiter= */ _config->delimiter, /* parallel= */ _parallel,
        /* hash_range= */ _config->hash_range);
    return processor;
  }

  /**
   * The Unlabeled non-updating processor is used for inference and
   * explanations. These processes should not update the history because the
   * tracked variable is often unavailable during inference. E.g. If we track
   * the movies watched by a user to recommend the next movie to watch, the true
   * movie that he ends up watching is not available during inference. Thus, we
   * should not update the history.
   */
  dataset::GenericBatchProcessorPtr makeUnlabeledNonUpdatingProcessor(
      const ColumnNumberMap& column_number_map) {
    auto processor = dataset::GenericBatchProcessor::make(
        buildInputBlocks(/* column_numbers= */ column_number_map,
                         /* should_update_history= */ false),
        /* label_blocks= */ {}, /* has_header= */ false,
        /* delimiter= */ _config->delimiter, /* parallel= */ _parallel,
        /* hash_range= */ _config->hash_range);
    return processor;
  }

  void verifyProcessorsAreInitialized() {
    if (!_unlabeled_non_updating_processor ||
        !_labeled_history_updating_processor) {
      throw std::invalid_argument("Attempted inference before training.");
    }
  }

  void verifyColumnNumberMapIsInitialized() {
    if (!_column_number_map) {
      throw std::invalid_argument("Attempted inference before training.");
    }
  }

  std::vector<dataset::BlockPtr> buildInputBlocks(
      const ColumnNumberMap& column_numbers, bool should_update_history) {
    std::vector<dataset::BlockPtr> blocks =
        FeatureComposer::makeNonTemporalFeatureBlocks(
            *_config, _temporal_relationships, column_numbers, _vectors_map,
            _text_pairgram_word_limit, _contextual_columns);

    if (_temporal_relationships.empty()) {
      return blocks;
    }

    auto temporal_feature_blocks = FeatureComposer::makeTemporalFeatureBlocks(
        *_config, _temporal_relationships, column_numbers, _vectors_map,
        *_context, should_update_history);

    blocks.insert(blocks.end(), temporal_feature_blocks.begin(),
                  temporal_feature_blocks.end());
    return blocks;
  }

  dataset::GenericBatchProcessor& getProcessor(bool should_update_history) {
    return should_update_history ? *_labeled_history_updating_processor
                                 : *_unlabeled_non_updating_processor;
  }

  std::vector<std::string_view> toVectorOfStringViews(const LineInput& input) {
    return dataset::ProcessorUtils::parseCsvRow(input, _config->delimiter);
  }

  std::vector<std::string_view> toVectorOfStringViews(const MapInput& input) {
    verifyColumnNumberMapIsInitialized();
    std::vector<std::string_view> string_view_input(
        _column_number_map->numCols());
    for (const auto& [col_name, val] : input) {
      string_view_input[_column_number_map->at(col_name)] =
          std::string_view(val.data(), val.length());
    }
    return string_view_input;
  }

  std::vector<std::string> lineInputBatchFromMapInputBatch(
      const MapInputBatch& input_maps) {
    std::vector<std::string> string_batch(input_maps.size());
    for (uint32_t i = 0; i < input_maps.size(); i++) {
      auto vals = toVectorOfStringViews(input_maps[i]);
      string_batch[i] = concatenateWithDelimiter(vals, _config->delimiter);
    }
    return string_batch;
  }

  static std::string concatenateWithDelimiter(
      const std::vector<std::string_view>& substrings, char delimiter) {
    if (substrings.empty()) {
      return "";
    }
    std::stringstream s;
    s << substrings[0];
    std::for_each(
        substrings.begin() + 1, substrings.end(),
        [&](const std::string_view& substr) { s << delimiter << substr; });
    return s.str();
  }

  UDTConfigPtr _config;
  TemporalRelationships _temporal_relationships;

  TemporalContextPtr _context;
  std::unordered_map<std::string, dataset::ThreadSafeVocabularyPtr> _vocabs;
  PreprocessedVectorsMap _vectors_map;

  ColumnNumberMapPtr _column_number_map;
  std::vector<std::string> _column_number_to_name;

  /*
    The labeled history-updating processor is used for training and evaluation,
    which automatically updates the temporal context, as well as for manually
    updating the temporal context.

    The Unlabeled non-updating processor is used for inference and explanations.
    These processes should not update the history because the tracked variable
    is often unavailable during inference. E.g. if we track the movies watched
    by a user to recommend the next movie to watch, the true movie that he ends
    up watching is not available during inference, so we should not update the
    history.
  */
  dataset::GenericBatchProcessorPtr _labeled_history_updating_processor;
  dataset::GenericBatchProcessorPtr _unlabeled_non_updating_processor;

  uint32_t _input_dim;
  uint32_t _label_dim;
  bool _parallel;
  uint32_t _text_pairgram_word_limit;
  bool _contextual_columns;

  // Private constructor for cereal.
  UDTDatasetFactory() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactory>(this), _config,
            _temporal_relationships, _context, _vocabs, _vectors_map,
            _column_number_map, _column_number_to_name,
            _labeled_history_updating_processor,
            _unlabeled_non_updating_processor, _input_dim, _label_dim,
            _parallel, _text_pairgram_word_limit, _contextual_columns);
  }
};

using UDTDatasetFactoryPtr = std::shared_ptr<UDTDatasetFactory>;

class UDTDatasetFactoryConfig final : public DatasetLoaderFactoryConfig {
 public:
  explicit UDTDatasetFactoryConfig(
      HyperParameterPtr<UDTConfigPtr> config, HyperParameterPtr<bool> parallel,
      HyperParameterPtr<uint32_t> text_pairgram_word_limit,
      HyperParameterPtr<bool> contextual_columns)
      : _config(std::move(config)),
        _parallel(std::move(parallel)),
        _text_pairgram_word_limit(std::move(text_pairgram_word_limit)),
        _contextual_columns(std::move(contextual_columns)) {}

  DatasetLoaderFactoryPtr createDatasetState(
      const UserInputMap& user_specified_parameters) const final {
    auto config = _config->resolve(user_specified_parameters);
    auto parallel = _parallel->resolve(user_specified_parameters);
    auto text_pairgram_word_limit =
        _text_pairgram_word_limit->resolve(user_specified_parameters);

    return UDTDatasetFactory::make(config, parallel, text_pairgram_word_limit);
  }

 private:
  HyperParameterPtr<UDTConfigPtr> _config;
  HyperParameterPtr<bool> _parallel;
  HyperParameterPtr<uint32_t> _text_pairgram_word_limit;
  HyperParameterPtr<bool> _contextual_columns;

  // Private constructor for cereal.
  UDTDatasetFactoryConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactoryConfig>(this), _config,
            _parallel, _text_pairgram_word_limit, _contextual_columns);
  }
};

}  // namespace thirdai::automl::deployment

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::UDTDatasetFactoryConfig)

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::UDTDatasetFactory)