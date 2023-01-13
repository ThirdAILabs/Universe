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
#include <dataset/src/DataSource.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
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

using dataset::ColumnNumberMap;
class UDTDatasetFactory;
using UDTDatasetFactoryPtr = std::shared_ptr<UDTDatasetFactory>;

class UDTDatasetFactory final : public DatasetLoaderFactory {
 public:
  explicit UDTDatasetFactory(UDTConfigPtr config, bool force_parallel,
                             uint32_t text_pairgram_word_limit,
                             bool contextual_columns = false,
                             std::optional<dataset::RegressionBinningStrategy>
                                 regression_binning = std::nullopt);

  static std::shared_ptr<UDTDatasetFactory> make(
      UDTConfigPtr config, bool force_parallel,
      uint32_t text_pairgram_word_limit, bool contextual_columns = false,
      std::optional<dataset::RegressionBinningStrategy> regression_binning =
          std::nullopt) {
    return std::make_shared<UDTDatasetFactory>(
        std::move(config), force_parallel, text_pairgram_word_limit,
        contextual_columns, regression_binning);
  }

  dataset::DatasetLoaderPtr getLabeledDatasetLoader(
      dataset::DataSourcePtr data_source, bool training) final;

  std::vector<BoltVector> featurizeInput(const LineInput& input) final {
    dataset::CsvSampleRef input_ref(input, _config->delimiter);
    return {getProcessor(/* should_update_history= */ false)
                .makeInputVector(input_ref)};
  }

  std::vector<BoltVector> featurizeInput(const MapInput& input) final {
    dataset::MapSampleRef input_ref(input);
    return {getProcessor(/* should_update_history= */ false)
                .makeInputVector(input_ref)};
  }

  std::vector<BoltVector> updateTemporalTrackers(const LineInput& input) {
    dataset::CsvSampleRef input_ref(input, _config->delimiter);
    return {getProcessor(/* should_update_history= */ true)
                .makeInputVector(input_ref)};
  }

  std::vector<BoltVector> updateTemporalTrackers(const MapInput& input) {
    dataset::MapSampleRef input_ref(input);
    return {getProcessor(/* should_update_history= */ true)
                .makeInputVector(input_ref)};
  }

  void updateMetadata(const std::string& col_name, const MapInput& update);

  void updateMetadataBatch(const std::string& col_name,
                           const MapInputBatch& updates);

  std::vector<BoltBatch> featurizeInputBatch(
      const LineInputBatch& inputs) final {
    return getProcessor(/* should_update_history= */ false).createBatch(inputs);
  }

  std::vector<BoltBatch> featurizeInputBatch(
      const MapInputBatch& inputs) final {
    dataset::MapBatchRef inputs_ref(inputs);
    return featurizeInputBatchImpl(inputs_ref,
                                   /* should_update_history= */ false);
  }

  uint32_t labelToNeuronId(std::variant<uint32_t, std::string> label) final;

  std::string className(uint32_t neuron_id) const final;

  std::vector<BoltBatch> batchUpdateTemporalTrackers(
      const LineInputBatch& inputs) {
    return getProcessor(/* should_update_history= */ true).createBatch(inputs);
  }

  std::vector<BoltBatch> batchUpdateTemporalTrackers(
      const MapInputBatch& inputs) {
    dataset::MapBatchRef inputs_ref(inputs);
    return featurizeInputBatchImpl(inputs_ref,
                                   /* should_update_history= */ true);
  }

  void resetTemporalTrackers() { _context->reset(); }

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio,
      const LineInput& sample) final {
    dataset::CsvSampleRef sample_ref(sample, _config->delimiter);
    return bolt::getSignificanceSortedExplanations(
        gradients_indices, gradients_ratio, sample_ref,
        _unlabeled_non_updating_processor);
  }

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio, const MapInput& sample) final {
    dataset::MapSampleRef sample_ref(sample);
    return bolt::getSignificanceSortedExplanations(
        gradients_indices, gradients_ratio, sample_ref,
        _unlabeled_non_updating_processor);
  }

  std::vector<bolt::InputPtr> getInputNodes() final {
    return {
        bolt::Input::make(_labeled_history_updating_processor->getInputDim())};
  }

  uint32_t getLabelDim() final {
    return _labeled_history_updating_processor->getLabelDim();
  }

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static UDTDatasetFactoryPtr load(const std::string& filename);

  static UDTDatasetFactoryPtr load_stream(std::istream& input_stream);

  void verifyCanDistribute();

  bool hasTemporalTracking() const final {
    return !_temporal_relationships.empty();
  }

  UDTConfigPtr config() { return _config; }

  void enableTargetCategoryNormalization() {
    _normalize_target_categories = true;
  }

 private:
  PreprocessedVectorsMap processAllMetadata();

  dataset::PreprocessedVectorsPtr makeProcessedVectorsForCategoricalColumn(
      const std::string& col_name, const CategoricalDataTypePtr& categorical);

  static ColumnNumberMap makeColumnNumberMapFromHeader(
      dataset::DataSource& data_source, char delimiter);

  std::vector<dataset::BlockPtr> buildMetadataInputBlocks(
      const CategoricalMetadataConfig& metadata_config) const;

  static dataset::PreprocessedVectorsPtr preprocessedVectorsFromDataset(
      dataset::DatasetLoader& dataset_loader,
      dataset::ThreadSafeVocabulary& key_vocab);

  std::vector<BoltBatch> featurizeInputBatchImpl(
      dataset::ColumnarInputBatch& inputs, bool should_update_history) {
    auto batches = getProcessor(should_update_history).createBatch(inputs);

    // We cannot use the initializer list because the copy constructor is
    // deleted for BoltBatch.
    std::vector<BoltBatch> batch_list;
    batch_list.emplace_back(std::move(batches.at(0)));
    return batch_list;
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

  void verifyColumnMetadataExists(const std::string& col_name) {
    if (!_config->data_types.count(col_name) ||
        !asCategorical(_config->data_types.at(col_name)) ||
        !asCategorical(_config->data_types.at(col_name))->metadata_config ||
        !_metadata_processors.count(col_name) ||
        !_vectors_map.count(col_name)) {
      throw std::invalid_argument("'" + col_name + "' is an invalid column.");
    }
  }

  auto getColumnMetadataConfig(const std::string& col_name) {
    return asCategorical(_config->data_types.at(col_name))->metadata_config;
  }

  std::vector<dataset::BlockPtr> buildInputBlocks(bool should_update_history);

  dataset::GenericBatchProcessor& getProcessor(bool should_update_history) {
    return should_update_history ? *_labeled_history_updating_processor
                                 : *_unlabeled_non_updating_processor;
  }

  static std::string concatenateWithDelimiter(
      const std::vector<std::string_view>& substrings, char delimiter);

  TemporalRelationships _temporal_relationships;
  UDTConfigPtr _config;

  TemporalContextPtr _context;
  std::unordered_map<std::string, dataset::ThreadSafeVocabularyPtr> _vocabs;

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
  std::unordered_map<std::string, dataset::GenericBatchProcessorPtr>
      _metadata_processors;

  bool _parallel;
  uint32_t _text_pairgram_word_limit;
  bool _contextual_columns;
  bool _normalize_target_categories;

  std::optional<dataset::RegressionBinningStrategy> _regression_binning;

  PreprocessedVectorsMap _vectors_map;
  dataset::GenericBatchProcessorPtr _labeled_history_updating_processor;
  dataset::GenericBatchProcessorPtr _unlabeled_non_updating_processor;

  // Private constructor for cereal.
  UDTDatasetFactory() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactory>(this), _config,
            _temporal_relationships, _context, _vocabs, _vectors_map,
            _column_number_to_name, _labeled_history_updating_processor,
            _unlabeled_non_updating_processor, _metadata_processors, _parallel,
            _text_pairgram_word_limit, _contextual_columns,
            _normalize_target_categories, _regression_binning);
  }
};

}  // namespace thirdai::automl::data
