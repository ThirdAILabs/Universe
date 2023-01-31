#pragma once

#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/dataset_factories/DatasetFactory.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/FeatureComposer.h>
#include <auto_ml/src/dataset_factories/udt/GraphConfig.h>
#include <auto_ml/src/dataset_factories/udt/UDTDatasetFactory.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/blocks/TabularHashFeatures.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <dataset/src/utils/PreprocessedVectors.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
namespace thirdai::automl::data {

class GraphDatasetFactory : public DatasetLoaderFactory {
  static constexpr const uint32_t DEFAULT_BATCH_SIZE = 2048;

 public:
  explicit GraphDatasetFactory(GraphConfigPtr conifg)
      : _config(std::move(conifg)) {}

  void prepareTheBatchProcessor() {
    auto input_blocks = buildInputBlocks();

    auto label_block = getLabelBlock();
    auto data_loader = dataset::FileDataSource::make(_config->_graph_file_name);

    _column_number_map = UDTDatasetFactory::makeColumnNumberMapFromHeader(
        *data_loader, _config->_delimeter);
    if (_config->_neighbourhood_context || _config->_label_context ||
        _config->_kth_neighbourhood) {
      _vectors = makeFinalProcessedVectors(*data_loader);

      auto graph_block = dataset::GraphCategoricalBlock::make(
          _config->_source, _vectors, _neighbours, _node_id_map);

      input_blocks.push_back(graph_block);
    }

    _batch_processor = dataset::TabularFeaturizer::make(
        /* input_blocks= */ std::move(input_blocks),
        /* label_blocks= */ {std::move(label_block)},
        /* has_header= */ true, /* delimiter= */ _config->_delimeter,
        /* parallel= */ true, /* hash_range= */ 100000);
  }

  std::vector<uint32_t> getInputDims() final {
    return {_batch_processor->getInputDim()};
  }

  dataset::DatasetLoaderPtr getLabeledDatasetLoader(
      std::shared_ptr<dataset::DataSource> data_source, bool training) final {
    _column_number_map = UDTDatasetFactory::makeColumnNumberMapFromHeader(
        *data_source, _config->_delimeter);

    // The batch processor will treat the next line as a header
    // Restart so batch processor does not skip a sample.
    data_source->restart();

    _batch_processor->updateColumnNumbers(_column_number_map);

    return std::make_unique<dataset::DatasetLoader>(data_source,
                                                    _batch_processor,
                                                    /* shuffle= */ training);
  }

  std::vector<BoltVector> featurizeInput(const LineInput& input) final {
    dataset::CsvSampleRef input_ref(input, _config->_delimeter);
    return {_batch_processor->makeInputVector(input_ref)};
  }

  std::vector<BoltBatch> featurizeInputBatch(
      const LineInputBatch& inputs) final {
    std::vector<BoltBatch> result;
    for (auto& batch : _batch_processor->featurize(inputs)) {
      result.emplace_back(std::move(batch));
    }
    return result;
  }

  uint32_t labelToNeuronId(std::variant<uint32_t, std::string> label) final {
    if (std::holds_alternative<uint32_t>(label)) {
      throw std::invalid_argument("Received an integer label");
    }
    const std::string& label_str = std::get<std::string>(label);
    return _target_vocab->getUid(label_str);
  }

  uint32_t getLabelDim() final { return _batch_processor->getLabelDim(); }

  dataset::TabularFeaturizerPtr getBatchProcessor() { return _batch_processor; }

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio,
      const std::string& sample) final {
    dataset::CsvSampleRef input(sample, _config->_delimeter);
    return bolt::getSignificanceSortedExplanations(
        gradients_indices, gradients_ratio, input, _batch_processor);
  }

  bool hasTemporalTracking() const final { return false; }

  static std::vector<std::vector<uint32_t>> createGraph(  // upto now correct
      const std::vector<std::vector<std::string>>& rows,
      const std::vector<uint32_t>& relationship_col_nums) {
    std::vector<std::vector<uint32_t>> adjacency_list_simulation(rows.size());
#pragma omp parallel for default(none) \
    shared(relationship_col_nums, rows, adjacency_list_simulation)
    for (uint32_t i = 0; i < rows.size(); i++) {
      for (uint32_t j = i + 1; j < rows.size(); j++) {
        for (unsigned int relationship_col_num : relationship_col_nums) {
          if (rows[i][relationship_col_num] == rows[j][relationship_col_num]) {
            adjacency_list_simulation[i].push_back(j);
            adjacency_list_simulation[j].push_back(i);
            break;
          }
        }
      }
    }
    return adjacency_list_simulation;
  }

  static std::vector<std::unordered_set<uint32_t>> findNeighboursForAllNodes(
      uint32_t num_nodes,
      const std::vector<std::vector<uint32_t>>& adjacency_list,
      uint32_t k_hop) {
    std::vector<std::unordered_set<uint32_t>> neighbours(num_nodes);
    for (uint32_t i = 0; i < num_nodes; i++) {
      std::unordered_set<uint32_t> neighbours_for_node;
      std::vector<bool> visited(num_nodes, false);
      findAllNeighboursForNode(k_hop, i, visited, neighbours_for_node,
                               adjacency_list);
      neighbours[i] = neighbours_for_node;
    }
    return neighbours;
  }

  static std::vector<std::vector<std::string>> processNumerical(
      const std::vector<std::vector<std::string>>& rows,
      const std::vector<uint32_t>& numerical_columns,
      const std::vector<std::unordered_set<uint32_t>>& neighbours) {
    std::vector<std::vector<std::string>> processed_numerical_columns(
        rows.size(), std::vector<std::string>(numerical_columns.size()));
#pragma omp parallel for default(none) \
    shared(rows, numerical_columns, processed_numerical_columns, neighbours)
    for (uint32_t i = 0; i < rows.size(); i++) {
      for (uint32_t j = 0; j < numerical_columns.size(); j++) {
        int value = 0;
        if (!neighbours[i].empty()) {
          for (auto neighbour : neighbours[i]) {
            value += stoi(rows[neighbour][numerical_columns[j]]);
          }
          processed_numerical_columns[i][j] =
              std::to_string(static_cast<float>(value) / neighbours[i].size());
        } else {
          processed_numerical_columns[i][j] = "0";
        }
      }
    }

    return processed_numerical_columns;
  }

 private:
  static std::vector<uint32_t> getRelationshipColumns(
      const std::vector<std::string>& columns,
      const dataset::ColumnNumberMap& column_number_map) {
    std::vector<uint32_t> relationship_col_nums(columns.size());
    for (const auto& column : columns) {
      relationship_col_nums.push_back(column_number_map.at(column));
    }
    return relationship_col_nums;
  }
  dataset::CsvRolledBatch getFinalData(
      const std::vector<std::vector<std::string>>& rows,
      const std::vector<uint32_t>& numerical_columns) {
    std::vector<uint32_t> relationship_col_nums = getRelationshipColumns(
        _config->_relationship_columns, _column_number_map);

    _adjacency_list = createGraph(rows, relationship_col_nums);

    _neighbours = findNeighboursForAllNodes(rows.size(), _adjacency_list,
                                            _config->_kth_neighbourhood);

    auto values = processNumerical(rows, numerical_columns, _neighbours);

    auto copied_rows = rows;

    for (uint32_t i = 0; i < rows.size(); i++) {
      copied_rows[i].insert(copied_rows[i].end(), values[i].begin(),
                            values[i].end());
    }

    dataset::CsvRolledBatch input(copied_rows);

    return input;
  }

  std::vector<std::vector<std::string>> getRawData(
      dataset::DataSource& data_loader) {
    uint32_t source_col_num = _column_number_map.at(_config->_source);

    std::vector<std::string> full_data;

    while (auto data = data_loader.nextBatch(DEFAULT_BATCH_SIZE)) {
      full_data.insert(full_data.end(), data->begin(), data->end());
    }

    std::vector<std::vector<std::string>> rows(full_data.size());

    std::vector<std::string> nodes(full_data.size());

    // #pragma omp parallel for default(none)
    //     shared(rows, full_data, nodes, source_col_num)
    for (uint32_t i = 0; i < full_data.size(); i++) {
      auto temp = dataset::ProcessorUtils::parseCsvRow(full_data[i],
                                                       _config->_delimeter);
      rows[i] = std::vector<std::string>(temp.begin(), temp.end());

      nodes[i] = rows[i][source_col_num];
    }

    _node_id_map = ColumnNumberMap(nodes);

    return rows;
  }

  dataset::PreprocessedVectorsPtr makeFinalProcessedVectors(
      dataset::DataSource& data_loader) {
    auto rows = getRawData(data_loader);
    std::vector<uint32_t> numerical_columns;
    std::vector<dataset::BlockPtr> input_blocks;
    std::vector<dataset::TabularColumn> tabular_columns;
    std::vector<data::NumericalDataTypePtr> numerical_types;
    uint32_t original_num_cols = _column_number_map.numCols();

    for (const auto& [col_name, data_type] : _config->_data_types) {
      uint32_t col_num = _column_number_map.at(col_name);
      if ((_config->_kth_neighbourhood && col_name != _config->_target) ||
          (_config->_label_context && col_name == _config->_target)) {
        if (auto categorical = asCategorical(data_type)) {
          if (categorical->delimiter) {
            input_blocks.push_back(dataset::NGramTextBlock::make(
                col_num, /* dim= */ 1, std::numeric_limits<uint32_t>::max(),
                *categorical->delimiter));
          } else {
            tabular_columns.push_back(dataset::TabularColumn::Categorical(
                /* identifier= */ col_num));
          }
        }
        if (auto text_meta = asText(data_type)) {
          if (text_meta->contextual_encoding == TextEncodingType::Pairgrams ||
              (text_meta->average_n_words &&
               text_meta->average_n_words <= 15)) {
            // text hash range of MAXINT is fine since features are later
            // hashed into a range. In fact it may reduce hash collisions.
            input_blocks.push_back(dataset::PairGramTextBlock::make(
                col_name, /* dim= */ std::numeric_limits<uint32_t>::max()));
          } else if (text_meta->contextual_encoding ==
                     TextEncodingType::Bigrams) {
            input_blocks.push_back(dataset::NGramTextBlock::make(
                col_name, /* n= */ 2,
                /* dim= */ std::numeric_limits<uint32_t>::max()));
          } else {
            input_blocks.push_back(dataset::NGramTextBlock::make(
                col_name, /* n= */ 1,
                /* dim= */ std::numeric_limits<uint32_t>::max()));
          }
        }
      }
      if (_config->_neighbourhood_context) {
        if (auto numerical = asNumerical(data_type)) {
          numerical_columns.push_back(col_num);
          numerical_types.push_back(asNumerical(data_type));
        }
      }
    }

    if (_config->_neighbourhood_context) {
      for (uint32_t i = 0; i < numerical_types.size(); i++) {
        tabular_columns.push_back(dataset::TabularColumn::Numeric(
            /* identifier= */ original_num_cols + i,
            /* range= */ numerical_types[i]->range,
            /* num_bins= */
            FeatureComposer::getNumberOfBins(numerical_types[i]->granularity)));
      }
    }
    input_blocks.push_back(std::make_shared<dataset::TabularHashFeatures>(
        /* columns= */ tabular_columns,
        /* output_range= */ std::numeric_limits<uint32_t>::max(),
        /* with_pairgrams= */ true));
    auto key_vocab = dataset::ThreadSafeVocabulary::make(
        /* vocab_size= */ 0, /* limit_vocab_size= */ false);
    auto label_block = dataset::StringLookupCategoricalBlock::make(
        _column_number_map.at(_config->_source), key_vocab);

    auto processor = dataset::TabularFeaturizer::make(
        /* input_blocks= */ std::move(input_blocks),
        /* label_blocks= */ {std::move(label_block)},
        /* has_header= */ false, /* delimiter= */ _config->_delimeter,
        /* parallel= */ true, /* hash_range= */ 100000);
    auto final_data = getFinalData(rows, numerical_columns);
    return makePreprocessedVectors(processor, *key_vocab, final_data);
  }

  std::vector<dataset::BlockPtr> buildInputBlocks() {
    UDTConfig feature_config(
        /* data_types= */ _config->_data_types,
        /* temporal_tracking_relationships= */ {},
        /* target= */ _config->_target,
        /* n_target_classes= */ _config->_n_target_classes);

    TemporalRelationships empty_temporal_relationships;

    PreprocessedVectorsMap empty_vectors_map;

    return FeatureComposer::makeNonTemporalFeatureBlocks(
        feature_config, empty_temporal_relationships, empty_vectors_map, 5,
        false);
  }

  dataset::BlockPtr getLabelBlock() {
    if (!_target_vocab) {
      _target_vocab = dataset::ThreadSafeVocabulary::make(
          /* vocab_size= */ _config->_n_target_classes);
    }
    return dataset::StringLookupCategoricalBlock::make(_config->_target,
                                                       _target_vocab);
  }

  static void findAllNeighboursForNode(  // NOLINT
      uint32_t k_hop, uint32_t node_id, std::vector<bool>& visited,
      std::unordered_set<uint32_t>& neighbours,
      const std::vector<std::vector<uint32_t>>& adjacency_list) {
    if (k_hop == 0) {
      return;
    }
    visited[node_id] = true;

    for (const auto& neighbour : adjacency_list[node_id]) {
      if (!visited[neighbour]) {
        neighbours.insert(neighbour);
        findAllNeighboursForNode(k_hop - 1, neighbour, visited, neighbours,
                                 adjacency_list);
      }
    }
  }

  static dataset::PreprocessedVectorsPtr makePreprocessedVectors(
      const dataset::TabularFeaturizerPtr& processor,
      dataset::ThreadSafeVocabulary& key_vocab, dataset::CsvRolledBatch rows) {
    auto batches = processor->featurize(rows);

    std::unordered_map<std::string, BoltVector> preprocessed_vectors(
        rows.size());

    for (uint32_t vec = 0; vec < batches[0].size(); vec++) {
      auto id = batches[1][vec].active_neurons[0];
      auto key = key_vocab.getString(id);
      preprocessed_vectors[key] = std::move(batches[0][vec]);
    }

    return std::make_shared<dataset::PreprocessedVectors>(
        std::move(preprocessed_vectors), processor->getInputDim());
  }

  GraphConfigPtr _config;
  dataset::PreprocessedVectorsPtr _vectors;
  ColumnNumberMap _column_number_map;
  ColumnNumberMap _node_id_map;
  std::vector<std::vector<uint32_t>> _adjacency_list;
  std::vector<std::unordered_set<uint32_t>> _neighbours;
  dataset::TabularFeaturizerPtr _batch_processor;
  dataset::ThreadSafeVocabularyPtr _target_vocab;
};

using GraphDatasetFactoryPtr = std::shared_ptr<GraphDatasetFactory>;

}  // namespace thirdai::automl::data