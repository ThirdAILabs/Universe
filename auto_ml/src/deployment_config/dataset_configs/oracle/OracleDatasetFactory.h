#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include "Aliases.h"
#include "FeatureComposer.h"
#include "OracleConfig.h"
#include "TemporalContext.h"
#include "TemporalRelationshipsAutotuner.h"
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
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

class OracleDatasetFactory final : public DatasetLoaderFactory {
 public:
  explicit OracleDatasetFactory(OracleConfigPtr config, bool parallel,
                                uint32_t text_pairgram_word_limit,
                                bool column_contextualization = false)
      : _config(std::move(config)),
        _temporal_relationships(TemporalRelationshipsAutotuner::autotune(
            _config->data_types, _config->provided_relationships,
            _config->lookahead)),
        _context(std::make_shared<TemporalContext>()),
        _parallel(parallel),
        _text_pairgram_word_limit(text_pairgram_word_limit),
        _column_contextualization(column_contextualization) {
    ColumnNumberMap mock_column_number_map(_config->data_types);
    auto mock_processor = makeLabeledUpdatingProcessor(mock_column_number_map);

    _input_dim = mock_processor->getInputDim();
    _label_dim = mock_processor->getLabelDim();
  }

  static std::shared_ptr<OracleDatasetFactory> make(
      OracleConfigPtr config, bool parallel, uint32_t text_pairgram_word_limit,
      bool column_contextualization = false) {
    return std::make_shared<OracleDatasetFactory>(std::move(config), parallel,
                                                  text_pairgram_word_limit,
                                                  column_contextualization);
  }

  DatasetLoaderPtr getLabeledDatasetLoader(
      std::shared_ptr<dataset::DataLoader> data_loader, bool training) final {
    auto header = data_loader->nextLine();
    if (!header) {
      throw std::invalid_argument(
          "The dataset must have a header that contains column names.");
    }

    auto current_column_number_map =
        std::make_shared<ColumnNumberMap>(*header, _config->delimiter);
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
      response.column_name = _column_number_to_name[response.column_number];
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
          /* col= */ col_num, /* n_classes= */ target_config.n_unique_classes,
          /* delimiter= */ target_config.delimiter);
    } else {
      if (!_vocabs.count(_config->target)) {
        _vocabs[_config->target] = dataset::ThreadSafeVocabulary::make(
            /* vocab_size= */ target_config.n_unique_classes);
      }
      label_block = dataset::StringLookupCategoricalBlock::make(
          /* col= */ col_num, /* vocab= */ _vocabs.at(_config->target),
          /* delimiter= */ target_config.delimiter);
    }

    auto input_blocks =
        buildInputBlocks(/* column_numbers= */ column_number_map,
                         /* should_update_history= */ true);

    auto processor = dataset::GenericBatchProcessor::make(
        std::move(input_blocks), {label_block});
    processor->setParallelism(_parallel);
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
        /* label_blocks= */ {});
    processor->setParallelism(_parallel);
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
            *_config, _temporal_relationships, column_numbers, _vocabs,
            _text_pairgram_word_limit, _column_contextualization);

    if (_temporal_relationships.empty()) {
      return blocks;
    }

    auto temporal_feature_blocks = FeatureComposer::makeTemporalFeatureBlocks(
        *_config, _temporal_relationships, column_numbers, _vocabs, *_context,
        should_update_history);

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

  OracleConfigPtr _config;
  TemporalRelationships _temporal_relationships;

  TemporalContextPtr _context;
  std::unordered_map<std::string, dataset::ThreadSafeVocabularyPtr> _vocabs;

  ColumnNumberMapPtr _column_number_map;
  std::unordered_map<uint32_t, std::string> _column_number_to_name;

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
  bool _column_contextualization;

  // Private constructor for cereal.
  OracleDatasetFactory() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactory>(this), _config,
            _temporal_relationships, _context, _vocabs, _column_number_map,
            _column_number_to_name, _labeled_history_updating_processor,
            _unlabeled_non_updating_processor, _input_dim, _label_dim,
            _parallel, _text_pairgram_word_limit, _column_contextualization);
  }
};

using OracleDatasetFactoryPtr = std::shared_ptr<OracleDatasetFactory>;

class OracleDatasetFactoryConfig final : public DatasetLoaderFactoryConfig {
 public:
  explicit OracleDatasetFactoryConfig(
      HyperParameterPtr<OracleConfigPtr> config,
      HyperParameterPtr<bool> parallel,
      HyperParameterPtr<uint32_t> text_pairgram_word_limit,
      HyperParameterPtr<bool> column_contextualization)
      : _config(std::move(config)),
        _parallel(std::move(parallel)),
        _text_pairgram_word_limit(std::move(text_pairgram_word_limit)),
        _column_contextualization(std::move(column_contextualization)) {}

  DatasetLoaderFactoryPtr createDatasetState(
      const UserInputMap& user_specified_parameters) const final {
    auto config = _config->resolve(user_specified_parameters);
    auto parallel = _parallel->resolve(user_specified_parameters);
    auto text_pairgram_word_limit =
        _text_pairgram_word_limit->resolve(user_specified_parameters);

    return OracleDatasetFactory::make(config, parallel,
                                      text_pairgram_word_limit);
  }

 private:
  HyperParameterPtr<OracleConfigPtr> _config;
  HyperParameterPtr<bool> _parallel;
  HyperParameterPtr<uint32_t> _text_pairgram_word_limit;
  HyperParameterPtr<bool> _column_contextualization;

  // Private constructor for cereal.
  OracleDatasetFactoryConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactoryConfig>(this), _config,
            _parallel, _text_pairgram_word_limit, _column_contextualization);
  }
};

}  // namespace thirdai::automl::deployment

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::OracleDatasetFactoryConfig)

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::OracleDatasetFactory)