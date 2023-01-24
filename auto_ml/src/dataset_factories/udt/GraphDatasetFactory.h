#pragma once

#include <auto_ml/src/dataset_factories/DatasetFactory.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/FeatureComposer.h>
#include <auto_ml/src/dataset_factories/udt/GraphConfig.h>
#include <auto_ml/src/dataset_factories/udt/UDTDatasetFactory.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/blocks/TabularHashFeatures.h>
#include <dataset/src/utils/PreprocessedVectors.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
namespace thirdai::automl::data {

class GraphDatasetFactory {
  static constexpr const uint32_t DEFAULT_BATCH_SIZE = 2048;

 public:
  explicit GraphDatasetFactory(GraphConfigPtr conifg)
      : _config(std::move(conifg)) {}

  void prepareTheBatchProcessor() {
    auto input_blocks = buildInputBlocks();

    auto label_block = getLabelBlock();
    if (_config->_neighbourhood_context || _config->_label_context ||
        _config->_kth_neighbourhood) {
      makePrepocessedProcessedVectors();

      auto graph_block = dataset::GraphCategoricalBlock::make(
          _config->_source, _vectors, _neighbours, _node_id_map);

      input_blocks.push_back(graph_block);
    }

    _batch_processor = dataset::GenericBatchProcessor::make(
        /* input_blocks= */ std::move(input_blocks),
        /* label_blocks= */ {std::move(label_block)},
        /* has_header= */ true, /* delimiter= */ _config->_delimeter,
        /* parallel= */ true, /* hash_range= */ 100000);
    _batch_processor->updateColumnNumbers(_column_number_map);
  }

  std::vector<uint32_t> getInputDim() {
    return {_batch_processor->getInputDim()};
  }

  uint32_t getLabelDim() { return _batch_processor->getLabelDim(); }

  dataset::GenericBatchProcessorPtr getBatchProcessor() {
    return _batch_processor;
  }

  static std::vector<std::vector<uint32_t>> createGraph(  // upto now correct
      const std::vector<std::vector<std::string>>& rows,
      const std::vector<uint32_t>& relationship_col_nums,
      uint32_t source_col_num) {
    std::vector<std::vector<uint32_t>> adjacency_list_simulation(rows.size());
    // #pragma omp parallel for default(none) shared(
    //     relationship_col_nums, rows, adjacency_list_simulation,
    //     source_col_num) collapse(2)
    for (uint32_t i = 0; i < rows.size(); i++) {
      for (uint32_t j = i + 1; j < rows.size(); j++) {
        if (rows[i][source_col_num] != rows[j][source_col_num]) {
          for (unsigned int relationship_col_num : relationship_col_nums) {
            if (rows[i][relationship_col_num] ==
                rows[j][relationship_col_num]) {
              adjacency_list_simulation[i].push_back(j);
              adjacency_list_simulation[j].push_back(i);
              break;
            }
          }
        }
      }
    }
    return adjacency_list_simulation;
  }

  std::vector<std::unordered_set<uint32_t>> findNeighboursForAllNodes(
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
    // #pragma omp parallel for default(none) shared(
    //     rows, k_hop, numerical_columns, processed_numerical_columns)
    for (uint32_t i = 0; i < rows.size(); i++) {
      for (uint32_t j = 0; j < numerical_columns.size(); j++) {
        int value = 0;
        for (auto neighbour : neighbours[i]) {
          value += stoi(rows[neighbour][numerical_columns[j]]);
        }
        processed_numerical_columns[i][j] =
            std::to_string(value / neighbours.size());
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
    uint32_t source_col_num = _column_number_map.at(_config->_source);
    std::vector<uint32_t> relationship_col_nums = getRelationshipColumns(
        _config->_relationship_columns, _column_number_map);

    _adjacency_list = createGraph(rows, relationship_col_nums, source_col_num);

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

  std::vector<std::vector<std::string>> getRawData() {
    auto data_loader = dataset::SimpleFileDataSource::make(
        _config->_graph_file_name,
        /* target_batch_size= */ DEFAULT_BATCH_SIZE);

    _column_number_map = UDTDatasetFactory::makeColumnNumberMapFromHeader(
        *data_loader, _config->_delimeter);

    uint32_t source_col_num = _column_number_map.at(_config->_source);

    std::vector<std::string> full_data;

    while (auto data = data_loader->nextBatch()) {
      full_data.insert(full_data.end(), data->begin(), data->end());
    }

    std::vector<std::vector<std::string>> rows;

    std::unordered_map<std::string, uint32_t> nodes;

    // #pragma omp parallel for default(none)
    //     shared(rows, full_data, nodes, source_col_num)
    for (uint32_t i = 0; i < full_data.size(); i++) {
      auto temp = dataset::ProcessorUtils::parseCsvRow(full_data[i],
                                                       _config->_delimeter);
      std::vector<std::string> v(temp.begin(), temp.end());
      rows.push_back(v);

      nodes[v[source_col_num]] = i;
    }

    _node_id_map = ColumnNumberMap(nodes);

    return rows;
  }

  void makePrepocessedProcessedVectors() {
    auto rows = getRawData();
    std::vector<dataset::BlockPtr> input_blocks;
    std::vector<dataset::TabularColumn> tabular_columns;
    std::vector<data::NumericalDataTypePtr> numerical_types;
    std::vector<uint32_t> numerical_columns;
    uint32_t original_num_cols = _column_number_map.numCols();

    for (const auto& [col_name, data_type] : _config->_data_types) {
      uint32_t col_num = _column_number_map.at(col_name);
      if ((_config->_kth_neighbourhood && col_name != _config->_target) ||
          (_config->_label_context && col_name == _config->_target)) {
        if (auto categorical = asCategorical(data_type)) {
          if (categorical->delimiter) {
            input_blocks.push_back(dataset::UniGramTextBlock::make(
                col_num, /* dim= */ std::numeric_limits<uint32_t>::max(),
                *categorical->delimiter));
          } else {
            tabular_columns.push_back(dataset::TabularColumn::Categorical(
                /* identifier= */ col_num));
          }
        }
      }
      if (_config->_neighbourhood_context &&
          data_type->data_type() == "numerical") {
        numerical_columns.push_back(col_num);
        numerical_types.push_back(asNumerical(data_type));
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
      input_blocks.push_back(std::make_shared<dataset::TabularHashFeatures>(
          /* columns= */ tabular_columns,
          /* output_range= */ std::numeric_limits<uint32_t>::max(),
          /* with_pairgrams= */ true));
    }
    auto key_vocab = dataset::ThreadSafeVocabulary::make(
        /* vocab_size= */ 0, /* limit_vocab_size= */ false);
    auto label_block = dataset::StringLookupCategoricalBlock::make(
        _column_number_map.at(_config->_source), key_vocab);

    auto processor = dataset::GenericBatchProcessor::make(
        /* input_blocks= */ std::move(input_blocks),
        /* label_blocks= */ {std::move(label_block)},
        /* has_header= */ false, /* delimiter= */ _config->_delimeter,
        /* parallel= */ true, /* hash_range= */ 100000);
    auto final_data = getFinalData(rows, numerical_columns);
    _vectors = makePreprocessedVectors(processor, *key_vocab, final_data);
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
    auto key_vocab = dataset::ThreadSafeVocabulary::make(
        /* vocab_size= */ _config->_n_target_classes);
    return dataset::StringLookupCategoricalBlock::make(_config->_target,
                                                       key_vocab);
  }

  void findAllNeighboursForNode(  // NOLINT
      uint32_t k_hop, uint32_t node_id, std::vector<bool>& visited,
      std::unordered_set<uint32_t>& neighbours,
      const std::vector<std::vector<uint32_t>>& adjacency_list) {
    if (k_hop == 0) {
      return;
    }

    for (const auto& neighbour : adjacency_list[node_id]) {
      if (!visited[neighbour]) {
        visited[neighbour] = true;
        neighbours.insert(neighbour);
        findAllNeighboursForNode(k_hop - 1, neighbour, visited, neighbours,
                                 adjacency_list);
      }
    }
  }

  static dataset::PreprocessedVectorsPtr makePreprocessedVectors(
      const dataset::GenericBatchProcessorPtr& processor,
      dataset::ThreadSafeVocabulary& key_vocab, dataset::CsvRolledBatch rows) {
    auto batches = processor->createBatch(rows);

    std::unordered_map<std::string, BoltVector> preprocessed_vectors(
        rows.size());

    for (uint32_t vec = 0; vec < batches[0].getBatchSize(); vec++) {
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
  dataset::GenericBatchProcessorPtr _batch_processor;
};

using GraphDatasetFactoryPtr = std::shared_ptr<GraphDatasetFactory>;

}  // namespace thirdai::automl::data