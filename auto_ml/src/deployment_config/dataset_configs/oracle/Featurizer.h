#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include "Aliases.h"
#include "FeatureComposer.h"
#include "OracleConfig.h"
#include "TemporalContext.h"
#include "TemporalRelationshipsAutotuner.h"
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::automl::deployment {

class Featurizer {
 public:
  explicit Featurizer(OracleConfigPtr config, bool parallel,
                      uint32_t text_pairgram_word_limit)
      : _config(std::move(config)),
        _temporal_relationships(TemporalRelationshipsAutotuner::autotune(
            _config->data_types, _config->provided_relationships,
            _config->lookahead)),
        _parallel(parallel),
        _text_pairgram_word_limit(text_pairgram_word_limit) {
    ColumnNumberMap mock_column_number_map(_config->data_types);
    TemporalContext mock_context(nullptr);
    auto mock_processor = makeLabeledContextUpdatingProcessor(
        mock_column_number_map, mock_context);

    _input_dim = mock_processor->getInputDim();
    _label_dim = mock_processor->getLabelDim();
  }

  void initializeProcessors(std::shared_ptr<dataset::DataLoader>& data_loader,
                            TemporalContext& context) {
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

    if (!_labeled_batch_processor || !_inference_batch_processor) {
      _labeled_batch_processor =
          makeLabeledContextUpdatingProcessor(*_column_number_map, context);
      _inference_batch_processor =
          makeUnlabeledNonUpdatingProcessor(*_column_number_map, context);
    }
  }

  std::vector<BoltVector> featurizeInput(const std::string& input,
                                         bool should_update_history) {
    auto row = stringInputToVectorOfStringViews(input, _config->delimiter);
    return featurizeInputImpl(row, should_update_history);
  }

  std::vector<BoltVector> featurizeInput(const MapInput& input,
                                         bool should_update_history) {
    verifyColumnNumberMapIsInitialized();
    auto row = vectorOfStringViewsFromMapInput(input, *_column_number_map);
    return featurizeInputImpl(row, should_update_history);
  }

  std::vector<BoltBatch> featurizeInputBatch(
      const std::vector<std::string>& inputs, bool should_update_history) {
    verifyProcessorsAreInitialized();

    auto& processor = should_update_history ? _labeled_batch_processor
                                            : _inference_batch_processor;

    auto [input_batch, _] = processor->createBatch(inputs);

    // We cannot use the initializer list because the copy constructor is
    // deleted for BoltBatch.
    std::vector<BoltBatch> batch_list;
    batch_list.emplace_back(std::move(input_batch));
    return batch_list;
  }

  std::vector<BoltBatch> featurizeInputBatch(const MapInputBatch& inputs,
                                             bool should_update_history) {
    verifyColumnNumberMapIsInitialized();

    auto string_batch = mapInputBatchToStringBatch(inputs, _config->delimiter,
                                                   *_column_number_map);

    return featurizeInputBatch(string_batch, should_update_history);
  }

  auto toInputRow(const std::string& sample) {
    return stringInputToVectorOfStringViews(sample, _config->delimiter);
  }

  auto toInputRow(const MapInput& sample) {
    return vectorOfStringViewsFromMapInput(sample, *_column_number_map);
  }

  uint32_t getInputDim() const { return _input_dim; }
  uint32_t getLabelDim() const { return _label_dim; }

  dataset::GenericBatchProcessorPtr getLabeledContextUpdatingProcessor() const {
    return _labeled_batch_processor;
  }

  dataset::GenericBatchProcessorPtr getUnlabeledNonUpdatingProcessor() const {
    return _inference_batch_processor;
  }

  std::string colNumToColName(uint32_t col_num) {
    return _column_number_to_name[col_num];
  }

 private:
  std::vector<BoltVector> featurizeInputImpl(
      std::vector<std::string_view>& input_row, bool should_update_history) {
    verifyProcessorsAreInitialized();
    auto& processor = should_update_history ? _labeled_batch_processor
                                            : _inference_batch_processor;
    BoltVector vector;
    if (auto exception = processor->makeInputVector(input_row, vector)) {
      std::rethrow_exception(exception);
    }
    return {std::move(vector)};
  }

  dataset::GenericBatchProcessorPtr makeLabeledContextUpdatingProcessor(
      const ColumnNumberMap& column_number_map, TemporalContext& context) {
    auto target_type = _config->data_types.at(_config->target);
    if (!target_type.isCategorical()) {
      throw std::invalid_argument(
          "Target column must be a categorical column.");
    }

    auto label_block = dataset::NumericalCategoricalBlock::make(
        /* col= */ column_number_map.at(_config->target),
        /* n_classes= */ target_type.asCategorical().n_unique_classes,
        /* delimiter= */ target_type.asCategorical().delimiter);

    auto input_blocks = buildInputBlocks(column_number_map, context,
                                         /* should_update_history= */ true);

    auto processor = dataset::GenericBatchProcessor::make(
        std::move(input_blocks), {label_block});
    processor->setParallelism(_parallel);
    return processor;
  }

  dataset::GenericBatchProcessorPtr makeUnlabeledNonUpdatingProcessor(
      const ColumnNumberMap& column_number_map, TemporalContext& context) {
    auto processor = dataset::GenericBatchProcessor::make(
        buildInputBlocks(column_number_map, context,
                         /* should_update_history= */ false),
        /* label_blocks= */ {});
    processor->setParallelism(_parallel);
    return processor;
  }

  void verifyProcessorsAreInitialized() {
    if (!_labeled_batch_processor || !_inference_batch_processor) {
      throw std::invalid_argument("Attempted inference before training.");
    }
  }

  void verifyColumnNumberMapIsInitialized() {
    if (!_column_number_map) {
      throw std::invalid_argument("Attempted inference before training.");
    }
  }

  std::vector<dataset::BlockPtr> buildInputBlocks(
      const ColumnNumberMap& column_numbers, TemporalContext& context,
      bool should_update_history) {
    std::vector<dataset::BlockPtr> blocks =
        FeatureComposer::makeNonTemporalFeatureBlocks(
            *_config, _temporal_relationships, column_numbers, _vocabs,
            _text_pairgram_word_limit);

    auto temporal_feature_blocks = FeatureComposer::makeTemporalFeatureBlocks(
        *_config, _temporal_relationships, column_numbers, _vocabs, context,
        should_update_history);

    blocks.insert(blocks.end(), temporal_feature_blocks.begin(),
                  temporal_feature_blocks.end());
    return blocks;
  }

  static std::vector<std::string_view> stringInputToVectorOfStringViews(
      const std::string& input_string, char delimiter) {
    return dataset::ProcessorUtils::parseCsvRow(input_string, delimiter);
  }

  static std::vector<std::string_view> vectorOfStringViewsFromMapInput(
      const MapInput& input_map, const ColumnNumberMap& column_number_map) {
    std::vector<std::string_view> string_view_input(
        column_number_map.numCols());
    for (const auto& [col_name, val] : input_map) {
      string_view_input[column_number_map.at(col_name)] =
          std::string_view(val.data(), val.length());
    }
    return string_view_input;
  }

  static std::vector<std::string> mapInputBatchToStringBatch(
      const MapInputBatch& input_maps, char delimiter,
      const ColumnNumberMap& column_number_map) {
    std::vector<std::string> string_batch(input_maps.size());
    for (uint32_t i = 0; i < input_maps.size(); i++) {
      auto vals =
          vectorOfStringViewsFromMapInput(input_maps[i], column_number_map);
      string_batch[i] = concatenateWithDelimiter(vals, delimiter);
    }
    return string_batch;
  }

  static std::string concatenateWithDelimiter(
      const std::vector<std::string_view>& substrings, char delimiter) {
    std::stringstream s;
    std::copy(substrings.begin(), substrings.end(),
              std::ostream_iterator<std::string_view>(s, &delimiter));
    return s.str();
  }

  OracleConfigPtr _config;
  TemporalRelationships _temporal_relationships;

  std::unordered_map<std::string, dataset::ThreadSafeVocabularyPtr> _vocabs;

  ColumnNumberMapPtr _column_number_map;
  std::unordered_map<uint32_t, std::string> _column_number_to_name;

  dataset::GenericBatchProcessorPtr _labeled_batch_processor;
  dataset::GenericBatchProcessorPtr _inference_batch_processor;

  uint32_t _input_dim;
  uint32_t _label_dim;
  bool _parallel;
  uint32_t _text_pairgram_word_limit;

  // Private constructor for cereal.
  Featurizer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_config, _temporal_relationships, _vocabs, _column_number_map,
            _column_number_to_name, _labeled_batch_processor,
            _inference_batch_processor, _input_dim, _label_dim, _parallel,
            _text_pairgram_word_limit);
  }
};

}  // namespace thirdai::automl::deployment