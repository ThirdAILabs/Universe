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
#include <auto_ml/src/dataset_factories/udt/CategoricalMetadata.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/ProcessorUtils.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
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

class UDTDatasetFactory;
using UDTDatasetFactoryPtr = std::shared_ptr<UDTDatasetFactory>;

class UDTDatasetFactory final : public DatasetLoaderFactory {
 public:
  explicit UDTDatasetFactory(const UDTConfigPtr& config, bool force_parallel,
                             uint32_t text_pairgram_word_limit,
                             bool contextual_columns = false,
                             std::optional<dataset::RegressionBinningStrategy>
                                 regression_binning = std::nullopt);

  static std::shared_ptr<UDTDatasetFactory> make(
      const UDTConfigPtr& config, bool force_parallel,
      uint32_t text_pairgram_word_limit, bool contextual_columns = false,
      std::optional<dataset::RegressionBinningStrategy> regression_binning =
          std::nullopt) {
    return std::make_shared<UDTDatasetFactory>(
        config, force_parallel, text_pairgram_word_limit, contextual_columns,
        regression_binning);
  }

  dataset::DatasetLoaderPtr getLabeledDatasetLoader(
      dataset::DataSourcePtr data_source, bool training) final;

  std::vector<BoltVector> featurizeInput(const MapInput& input) final {
    dataset::MapSampleRef input_ref(input);
    return {getProcessor(/* should_update_history= */ false)
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
      const MapInputBatch& inputs) final {
    dataset::MapBatchRef inputs_ref(inputs);
    return featurizeInputBatchImpl(inputs_ref,
                                   /* should_update_history= */ false);
  }

  uint32_t labelToNeuronId(std::variant<uint32_t, std::string> label) final;

  std::string className(uint32_t neuron_id) const final;

  std::vector<BoltBatch> batchUpdateTemporalTrackers(
      const MapInputBatch& inputs) {
    dataset::MapBatchRef inputs_ref(inputs);
    return featurizeInputBatchImpl(inputs_ref,
                                   /* should_update_history= */ true);
  }

  void resetTemporalTrackers() { _context->reset(); }

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio, const MapInput& sample) final {
    dataset::MapSampleRef sample_ref(sample);
    return bolt::getSignificanceSortedExplanations(
        gradients_indices, gradients_ratio, sample_ref,
        *_unlabeled_non_updating_processor);
  }

  std::vector<uint32_t> getInputDims() final {
    return {_labeled_history_updating_processor->getInputDim()};
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
  std::vector<BoltBatch> featurizeInputBatchImpl(
      dataset::ColumnarInputBatch& inputs, bool should_update_history) {
    auto batches = getProcessor(should_update_history).featurize(inputs);

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
  dataset::TabularFeaturizerPtr makeLabeledUpdatingProcessor();

  dataset::BlockPtr getLabelBlock();

  /**
   * The Unlabeled non-updating processor is used for inference and
   * explanations. These processes should not update the history because the
   * tracked variable is often unavailable during inference. E.g. If we track
   * the movies watched by a user to recommend the next movie to watch, the true
   * movie that he ends up watching is not available during inference. Thus, we
   * should not update the history.
   */
  dataset::TabularFeaturizerPtr makeUnlabeledNonUpdatingProcessor() {
    auto processor = dataset::TabularFeaturizer::make(
        buildInputBlocks(
            /* should_update_history= */ false),
        /* label_blocks= */ {}, /* has_header= */ true,
        /* delimiter= */ _config->delimiter, /* parallel= */ _parallel,
        /* hash_range= */ _config->hash_range);
    return processor;
  }

  std::vector<dataset::BlockPtr> buildInputBlocks(bool should_update_history);

  dataset::TabularFeaturizer& getProcessor(bool should_update_history) {
    return should_update_history ? *_labeled_history_updating_processor
                                 : *_unlabeled_non_updating_processor;
  }

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

  bool _parallel;
  uint32_t _text_pairgram_word_limit;
  bool _contextual_columns;
  bool _normalize_target_categories;

  std::optional<dataset::RegressionBinningStrategy> _regression_binning;

  CategoricalMetadata _categorical_metadata;

  dataset::TabularFeaturizerPtr _labeled_history_updating_processor;
  dataset::TabularFeaturizerPtr _unlabeled_non_updating_processor;

  // Private constructor for cereal.
  UDTDatasetFactory() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactory>(this), _config,
            _temporal_relationships, _context, _vocabs, _column_number_to_name,
            _categorical_metadata, _labeled_history_updating_processor,
            _unlabeled_non_updating_processor, _parallel,
            _text_pairgram_word_limit, _contextual_columns,
            _normalize_target_categories, _regression_binning);
  }
};

}  // namespace thirdai::automl::data
