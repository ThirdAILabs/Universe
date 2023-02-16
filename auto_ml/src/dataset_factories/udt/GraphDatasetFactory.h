#pragma once

#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/dataset_factories/DatasetFactory.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/FeatureComposer.h>
#include <auto_ml/src/dataset_factories/udt/GraphConfig.h>
#include <auto_ml/src/dataset_factories/udt/UDTDatasetFactory.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/DenseArray.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/blocks/TabularHashFeatures.h>
#include <dataset/src/featurizers/GraphFeaturizer.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <dataset/src/utils/PreprocessedVectors.h>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
namespace thirdai::automl::data {

using dataset::ColumnNumberMap;
using dataset::Neighbours;

class GraphDatasetFactory : public DatasetLoaderFactory {
 public:
  explicit GraphDatasetFactory(GraphConfigPtr config);

  dataset::GraphFeaturizerPtr prepareTheFeaturizer(
      const GraphConfigPtr& config,
      const std::vector<std::vector<std::string>>& rows);

  std::vector<uint32_t> getInputDims() final {
    return {_featurizer->getInputDim()};
  }

  dataset::DatasetLoaderPtr getLabeledDatasetLoader(
      std::shared_ptr<dataset::DataSource> data_source, bool training) final {
    return std::make_unique<dataset::DatasetLoader>(data_source, _featurizer,
                                                    /* shuffle= */ training);
  }

  std::vector<BoltVector> featurizeInput(const LineInput& /*input*/) final {
    throw std::invalid_argument("not implemeted yet!!");
  }

  std::vector<BoltBatch> featurizeInputBatch(
      const LineInputBatch& inputs) final {
    std::vector<BoltBatch> result;
    for (auto& batch : _featurizer->featurize(inputs)) {
      result.emplace_back(std::move(batch));
    }
    return result;
  }

  uint32_t labelToNeuronId(std::variant<uint32_t, std::string> label) final {
    if (std::holds_alternative<uint32_t>(label)) {
      if (!_config->_integer_target) {
        throw std::invalid_argument(
            "Received an integer but integer_target is set to False (it is "
            "False by default). Target must be passed "
            "in as a string.");
      }
      return std::get<uint32_t>(label);
    }
    const std::string& label_str = std::get<std::string>(label);
    return _target_vocab->getUid(label_str);
  }

  uint32_t getLabelDim() final { return _featurizer->getLabelDim(); }

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& /*gradients_indices*/,
      const std::vector<float>& /*gradients_ratio*/,
      const std::string& /*sample*/) final {
    throw std::invalid_argument("not implemeted yet");
  }

  bool hasTemporalTracking() const final { return false; }

  static std::pair<std::unordered_map<std::string, std::vector<std::string>>,
                   ColumnNumberMap>
  createGraph(const std::vector<std::vector<std::string>>& rows,
              const std::vector<uint32_t>& relationship_col_nums,
              uint32_t source_col_num);

  static Neighbours findNeighboursForAllNodes(
      const std::unordered_map<std::string, std::vector<std::string>>&
          adjacency_list,
      uint32_t k, const ColumnNumberMap& node_id_map);

 private:
  static std::vector<uint32_t> getRelationshipColumns(
      const std::vector<std::string>& columns,
      const dataset::ColumnNumberMap& column_number_map);

  static std::vector<std::vector<std::string>> processNumericalColumns(
      const std::vector<std::vector<std::string>>& rows,
      const std::vector<uint32_t>& numerical_columns,
      const Neighbours& neighbours, uint32_t source_col_num,
      const ColumnNumberMap& node_id_map);

  std::pair<std::unordered_map<std::string, std::vector<std::string>>,
            ColumnNumberMap>
  getGraphStructureInfo(const std::vector<std::vector<std::string>>& rows);

  dataset::CsvRolledBatch getFinalProcessedData(
      const std::vector<std::vector<std::string>>& rows,
      const std::vector<uint32_t>& numerical_columns,
      const ColumnNumberMap& node_id_map, const Neighbours& neighbours);

  std::vector<std::vector<std::string>> getCsvData(
      dataset::DataSource& data_loader);

  dataset::PreprocessedVectorsPtr makeNumericalProcessedVectors(
      const std::vector<std::vector<std::string>>& rows,
      const ColumnNumberMap& node_id_map, const Neighbours& neighbours);

  dataset::PreprocessedVectorsPtr makeFeatureProcessedVectors(
      const std::vector<std::vector<std::string>>& rows);

  static std::vector<dataset::BlockPtr> buildInputBlocks(
      const GraphConfigPtr& config);

  dataset::BlockPtr getLabelBlock(const GraphConfigPtr& config) {
    auto target_type = _config->_data_types.at(_config->_target);
    auto target_config = asCategorical(target_type);
    if (_config->_integer_target) {
      return dataset::NumericalCategoricalBlock::make(
          /* col= */ _config->_target,
          /* n_classes= */ _config->_n_target_classes,
          /* delimiter= */ target_config->delimiter);
    }
    if (!_target_vocab) {
      _target_vocab = dataset::ThreadSafeVocabulary::make(
          /* vocab_size= */ config->_n_target_classes);
    }
    return dataset::StringLookupCategoricalBlock::make(config->_target,
                                                       _target_vocab);
  }

  static void findAllNeighboursForNode(
      uint32_t k, const std::string& node_id, std::vector<bool>& visited,
      std::unordered_set<std::string>& neighbours,
      const std::unordered_map<std::string, std::vector<std::string>>&
          adjacency_list,
      const ColumnNumberMap& node_id_map);

  static dataset::PreprocessedVectorsPtr makePreprocessedVectors(
      const dataset::TabularFeaturizerPtr& processor,
      dataset::ThreadSafeVocabulary& key_vocab, dataset::CsvRolledBatch rows);

  GraphConfigPtr _config;
  ColumnNumberMap _column_number_map;
  dataset::GraphFeaturizerPtr _featurizer;
  dataset::ThreadSafeVocabularyPtr _target_vocab;
};

using GraphDatasetFactoryPtr = std::shared_ptr<GraphDatasetFactory>;

}  // namespace thirdai::automl::data