#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include "DataTypes.h"
#include "FeatureComposer.h"
#include "TemporalContext.h"
#include "TemporalRelationshipsAutotuner.h"
#include "UDTConfig.h"
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/dataset_factories/DatasetFactory.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <dataset/src/DataLoader.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
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

namespace thirdai::automl::data {

using dataset::ColumnNumberMapPtr;

class UDTDatasetFactory final : public DatasetLoaderFactory {
 public:
  explicit UDTDatasetFactory(UDTConfigPtr config, bool force_parallel,
                             uint32_t text_pairgram_word_limit,
                             bool contextual_columns = false,
                             std::optional<dataset::RegressionBinningStrategy>
                                 regression_binning = std::nullopt)
      : _config(std::move(config)),
        _temporal_relationships(TemporalRelationshipsAutotuner::autotune(
            _config->data_types, _config->provided_relationships,
            _config->lookahead)),
        _context(std::make_shared<TemporalContext>()),
        _parallel(_temporal_relationships.empty() || force_parallel),
        _text_pairgram_word_limit(text_pairgram_word_limit),
        _contextual_columns(contextual_columns),
        _regression_binning(regression_binning) {
    FeatureComposer::verifyConfigIsValid(*_config, _temporal_relationships);

    _vectors_map = processAllMetadata();

    _labeled_history_updating_processor = makeLabeledUpdatingProcessor();
    _unlabeled_non_updating_processor = makeUnlabeledNonUpdatingProcessor();

    _labeled_history_updating_processor->getInputDim();
    _labeled_history_updating_processor->getLabelDim();
  }

  static std::shared_ptr<UDTDatasetFactory> make(
      UDTConfigPtr config, bool force_parallel,
      uint32_t text_pairgram_word_limit, bool contextual_columns = false,
      std::optional<dataset::RegressionBinningStrategy> regression_binning =
          std::nullopt) {
    return std::make_shared<UDTDatasetFactory>(
        std::move(config), force_parallel, text_pairgram_word_limit,
        contextual_columns, regression_binning);
  }

  DatasetLoaderPtr getLabeledDatasetLoader(
      std::shared_ptr<dataset::DataLoader> data_loader, bool training) final;

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
    return featurizeInputBatchImpl(inputs, /* should_update_history= */ false);
  }

  uint32_t labelToNeuronId(std::variant<uint32_t, std::string> label) final;

  std::string className(uint32_t neuron_id) const final;

  std::vector<BoltBatch> batchUpdateTemporalTrackers(
      const LineInputBatch& inputs) {
    return featurizeInputBatchImpl(inputs, /* should_update_history= */ true);
  }

  std::vector<BoltBatch> batchUpdateTemporalTrackers(
      const MapInputBatch& inputs) {
    return featurizeInputBatchImpl(inputs, /* should_update_history= */ true);
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
  PreprocessedVectorsMap processAllMetadata();

  dataset::PreprocessedVectorsPtr makeProcessedVectorsForCategoricalColumn(
      const std::string& col_name, const CategoricalDataTypePtr& categorical);

  static ColumnNumberMapPtr makeColumnNumberMapFromHeader(
      dataset::DataLoader& data_loader, char delimiter);

  std::vector<dataset::BlockPtr> buildMetadataInputBlocks(
      const CategoricalMetadataConfig& metadata_config) const;

  static dataset::PreprocessedVectorsPtr preprocessedVectorsFromDataset(
      dataset::StreamingGenericDatasetLoader& dataset,
      dataset::ThreadSafeVocabulary& key_vocab);

  template <typename InputType>
  std::vector<BoltVector> featurizeInputImpl(const InputType& input,
                                             bool should_update_history) {
    verifyProcessorsAreInitialized();

    BoltVector vector;
    if (auto exception = getProcessor(should_update_history)
                             .makeInputVector(input, vector)) {
      std::rethrow_exception(exception);
    }
    return {std::move(vector)};
  }

  template <typename InputBatchType>
  std::vector<BoltBatch> featurizeInputBatchImpl(const InputBatchType& inputs,
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

    auto result = bolt::getSignificanceSortedExplanations(
        gradients_indices, gradients_ratio, sample,
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
  dataset::GenericBatchProcessorPtr makeLabeledUpdatingProcessor();

  dataset::BlockPtr getLabelBlock();

  /**
   * The Unlabeled non-updating processor is used for inference and
   * explanations. These processes should not update the history because the
   * tracked variable is often unavailable during inference. E.g. If we track
   * the movies watched by a user to recommend the next movie to watch, the true
   * movie that he ends up watching is not available during inference. Thus, we
   * should not update the history.
   */
  dataset::GenericBatchProcessorPtr makeUnlabeledNonUpdatingProcessor() {
    auto processor = dataset::GenericBatchProcessor::make(
        buildInputBlocks(
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

  std::vector<dataset::BlockPtr> buildInputBlocks(bool should_update_history);

  dataset::GenericBatchProcessor& getProcessor(bool should_update_history) {
    return should_update_history ? *_labeled_history_updating_processor
                                 : *_unlabeled_non_updating_processor;
  }

  static std::string concatenateWithDelimiter(
      const std::vector<std::string_view>& substrings, char delimiter);

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

  std::optional<dataset::RegressionBinningStrategy> _regression_binning;

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
            _parallel, _text_pairgram_word_limit, _contextual_columns,
            _regression_binning);
  }
};

using UDTDatasetFactoryPtr = std::shared_ptr<UDTDatasetFactory>;

}  // namespace thirdai::automl::data
