#pragma once

#include <auto_ml/src/dataset_factories/DatasetFactory.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/FeatureComposer.h>
#include <auto_ml/src/dataset_factories/udt/GraphConfig.h>
#include <auto_ml/src/dataset_factories/udt/UDTDatasetFactory.h>
#include <dataset/src/blocks/TabularHashFeatures.h>
#include <dataset/src/utils/PreprocessedVectors.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
namespace thirdai::automl::data {

class GraphDatasetFactory : public DatasetLoaderFactory {
 public:
  explicit GraphDatasetFactory(GraphConfigPtr conifg)
      : _config(std::move(conifg)) {}

 private:
  void processGraph() {
    auto data_loader = dataset::SimpleFileDataSource::make(
        _config->_graph_file_name,
        /* target_batch_size= */ _config->_batch_size);

    _column_number_map = UDTDatasetFactory::makeColumnNumberMapFromHeader(
        *data_loader, _config->_delimeter);

    auto data = data_loader->nextBatch();

    std::vector<std::vector<std::string_view>> rows(_config->_batch_size);

    std::vector<std::vector<uint32_t>> adjacency_list_simulation(
        _config->_batch_size);

    uint32_t source_col_num = _column_number_map.at(_config->_source);

    std::vector<std::string> nodes(rows.size());

#pragma omp parallel for default(none) shared(rows, data, nodes, source_col_num)
    for (uint32_t i = 0; i < _config->_batch_size; i++) {
      rows[i] = dataset::ProcessorUtils::parseCsvRow(data->at(i),
                                                     _config->_delimeter);

      nodes[i] = rows[i][source_col_num];
    }

    _node_id_map = nodes;

    std::vector<uint32_t> relationship_col_nums;
    for (const auto& column : _config->_relationship_columns) {
      relationship_col_nums.push_back(_column_number_map.at(column));
    }

#pragma omp parallel for default(none) shared(                              \
    relationship_col_nums, rows, adjacency_list_simulation, source_col_num) \
    collapse(2)
    for (uint32_t i = 0; i < _config->_batch_size; i++) {
      for (uint32_t j = i + 1; j < _config->_batch_size; j++) {
        for (unsigned int relationship_col_num : relationship_col_nums) {
          if (rows[i][relationship_col_num] == rows[j][relationship_col_num]) {
            adjacency_list_simulation[i].push_back(j);
            adjacency_list_simulation[j].push_back(i);
            break;
          }
        }
      }
    }

    _adjacency_list = adjacency_list_simulation;

    uint32_t original_num_cols = _column_number_map.numCols();

    std::vector<dataset::TabularColumn> tabular_columns;

    std::vector<data::NumericalDataTypePtr> numerical_types;

    std::vector<dataset::BlockPtr> blocks;

    std::vector<uint32_t> numerical_columns;

    if (_config->_kth_neighbourhood || _config->_neighbourhood_context ||
        _config->_label_context) {
      for (const auto& [col_name, data_type] : _config->_data_types) {
        uint32_t col_num = _column_number_map.at(col_name);
        if ((_config->_kth_neighbourhood && col_name != _config->_target) ||
            (_config->_label_context && col_name == _config->_target)) {
          if (auto categorical = asCategorical(data_type)) {
            if (categorical->delimiter) {
              blocks.push_back(dataset::UniGramTextBlock::make(
                  col_num, /* dim= */ std::numeric_limits<uint32_t>::max(),
                  *categorical->delimiter));
            } else {
              tabular_columns.push_back(dataset::TabularColumn::Categorical(
            /* identifier= */ col_name));
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
        auto values = processNumerical(rows, numerical_columns,
                                       _config->_kth_neighbourhood);
        for (uint32_t i = 0; i < rows.size(); i++) {
          rows[i].insert(rows[i].end(), values[i].begin(), values[i].end());
        }
        for (uint32_t i = 0; i < numerical_columns.size(); i++) {
        tabular_columns.push_back(dataset::TabularColumn::Numeric(
          /* identifier= */ original_num_cols+i, /* range= */ numerical_types[i]->range,
          /* num_bins= */ FeatureComposer::getNumberOfBins(numerical_types[i]->granularity)));
      }
      blocks.push_back(std::make_shared<dataset::TabularHashFeatures>(
      /* columns= */ tabular_columns,
      /* output_range= */ std::numeric_limits<uint32_t>::max(),
      /* with_pairgrams= */ true));
    }

    auto key_vocab = dataset::ThreadSafeVocabulary::make(
        /* vocab_size= */ 0, /* limit_vocab_size= */ false);
    auto label_block = dataset::StringLookupCategoricalBlock::make(
        _column_number_map.at(_config->_source), key_vocab);

    auto graph_processor = dataset::GenericBatchProcessor::make(
        /* input_blocks= */ std::move(blocks),
        /* label_blocks= */ {std::move(label_block)},
        /* has_header= */ false, /* delimiter= */ _config->_delimeter,
        /* parallel= */ true, /* hash_range= */ 100000);

    // auto [vectors, ids] = graph_processor->getBatch(rows);

    // std::unordered_map<std::string, BoltVector> preprocessed_vectors(
    //     rows.size());

    // for (uint32_t vec = 0; vec < vectors.getBatchSize(); vec++) {
    //   auto id = ids[vec].active_neurons[0];
    //   auto key = key_vocab->getString(id);
    //   preprocessed_vectors[key] = std::move(vectors[vec]);
    // }

    // _vectors = std::make_shared<dataset::PreprocessedVectors>(
    //     std::move(preprocessed_vectors), graph_processor->getInputDim());
  }
  }

  std::vector<std::vector<std::string>> processNumerical(
      const std::vector<std::vector<std::string_view>>& rows,
      const std::vector<uint32_t>& numerical_columns, uint32_t k_hop) {
    std::vector<std::vector<std::string>> processed_numerical_columns(
        rows.size(), std::vector<std::string>(numerical_columns.size()));
#pragma omp parallel for default(none) shared( \
    rows, k_hop, numerical_columns, processed_numerical_columns) collapse(2)
    for (uint32_t i = 0; i < rows.size(); i++) {
      std::unordered_set<uint32_t> neighbours;
      std::vector<bool> visited(rows.size(), false);
      findAllNeighbours(k_hop, i, visited, neighbours);
      for (uint32_t j = 0; j < numerical_columns.size(); j++) {
        uint32_t value = 0;
        for (unsigned int neighbour : neighbours) {
          value += stoi(std::string(rows[neighbour][numerical_columns[j]]));
        }
        processed_numerical_columns[i][j] =
            std::to_string(value / neighbours.size());
      }
    }

    return processed_numerical_columns;
  }

  void findAllNeighbours(uint32_t k_hop, uint32_t node_id,  // NOLINT
                         std::vector<bool>& visited,
                         std::unordered_set<uint32_t>& neighbours) {
    if (k_hop == 0) {
      return;
    }

    for (const auto& neighbour : _adjacency_list[node_id]) {
      if (!visited[neighbour]) {
        visited[neighbour] = true;
        neighbours.insert(neighbour);
        findAllNeighbours(k_hop - 1, neighbour, visited, neighbours);
      }
    }
  }

  void buildPreprocessedVectors() {}

  GraphConfigPtr _config;
  dataset::PreprocessedVectorsPtr _vectors;
  ColumnNumberMap _column_number_map;
  std::vector<std::string> _node_id_map;
  std::vector<std::vector<uint32_t>> _adjacency_list;
};

}  // namespace thirdai::automl::data