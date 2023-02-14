#include "GraphDatasetFactory.h"
#include <dataset/src/blocks/TabularHashFeatures.h>

namespace thirdai::automl::data {

GraphDatasetFactory::GraphDatasetFactory(GraphConfigPtr conifg)
    : _config(std::move(conifg)) {
  auto data_source = dataset::FileDataSource::make(_config->_graph_file_name);

  _column_number_map = UDTDatasetFactory::makeColumnNumberMapFromHeader(
      *data_source, _config->_delimeter);

  auto rows = getRawData(*data_source);

  _featurizer = prepareTheFeaturizer(_config, rows);
}

dataset::GraphFeaturizerPtr GraphDatasetFactory::prepareTheFeaturizer(
    const GraphConfigPtr& config,
    const std::vector<std::vector<std::string>>& rows) {
  std::vector<dataset::BlockPtr> input_blocks = buildInputBlocks(config);

  dataset::BlockPtr label_block = getLabelBlock(config);

  auto [adjacency_list, node_id_map] = getGraphStructureInfo(rows);

  std::unordered_map<std::string, std::unordered_set<std::string>> neighbours =
      findNeighboursForAllNodes(adjacency_list, config->_k_hop, node_id_map);

  if (config->_features_context) {
    dataset::PreprocessedVectorsPtr feature_vectors =
        makeFeatureProcessedVectors(rows);

    auto graph_block = dataset::GraphCategoricalBlock::make(
        config->_source, feature_vectors, neighbours);

    input_blocks.push_back(graph_block);
  }
  if (config->_numerical_context) {
    dataset::PreprocessedVectorsPtr numerical_vectors =
        makeNumericalProcessedVectors(rows, node_id_map, neighbours);

    auto graph_numerical_block = dataset::MetadataCategoricalBlock::make(
        config->_source, numerical_vectors);

    input_blocks.push_back(graph_numerical_block);
  }

  // TODO(YASH): remove the hard code of 100000 in hash range.
  auto featurizer = dataset::GraphFeaturizer::make(
      std::move(input_blocks), {std::move(label_block)}, config->_source,
      config->_max_neighbours, config->_delimeter, /*hash_range=*/100000);

  featurizer->updateNeighbours(neighbours);

  featurizer->updateNodeIdMap(node_id_map);

  return featurizer;
}

std::pair<std::unordered_map<std::string, std::vector<std::string>>,
          ColumnNumberMap>
GraphDatasetFactory::createGraph(
    const std::vector<std::vector<std::string>>& rows,
    const std::vector<uint32_t>& relationship_col_nums,
    uint32_t source_col_num) {
  std::unordered_map<std::string, std::vector<std::string>> adjacency_list;
  std::vector<std::string> nodes(rows.size());
  for (uint64_t row_1 = 0; row_1 < rows.size(); row_1++) {
    nodes[row_1] = rows[row_1][source_col_num];
    for (uint64_t row_2 = row_1 + 1; row_2 < rows.size(); row_2++) {
      for (unsigned int relationship_col_num : relationship_col_nums) {
        if (rows[row_1][relationship_col_num] ==
            rows[row_2][relationship_col_num]) {
          adjacency_list[rows[row_1][source_col_num]].push_back(
              rows[row_2][source_col_num]);
          adjacency_list[rows[row_2][source_col_num]].push_back(
              rows[row_1][source_col_num]);
          break;
        }
      }
    }
  }
  return {adjacency_list, ColumnNumberMap(nodes)};
}

std::unordered_map<std::string, std::unordered_set<std::string>>
GraphDatasetFactory::findNeighboursForAllNodes(
    const std::unordered_map<std::string, std::vector<std::string>>&
        adjacency_list,
    uint32_t k, const ColumnNumberMap& node_id_map) {
  uint32_t num_nodes = adjacency_list.size();
  std::unordered_map<std::string, std::unordered_set<std::string>> neighbours(
      num_nodes);
  for (const auto& temp : adjacency_list) {
    std::unordered_set<std::string> neighbours_for_node;
    std::vector<bool> visited(num_nodes, false);
    findAllNeighboursForNode(k, /*node_id=*/temp.first, visited,
                             neighbours_for_node, adjacency_list, node_id_map);
    neighbours[temp.first] = neighbours_for_node;
  }
  return neighbours;
}

std::vector<std::vector<std::string>> GraphDatasetFactory::processNumerical(
    const std::vector<std::vector<std::string>>& rows,
    const std::vector<uint32_t>& numerical_columns,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        neighbours,
    uint32_t source_col_num, const ColumnNumberMap& node_id_map) {
  std::vector<std::vector<std::string>> processed_numerical_columns(
      rows.size(), std::vector<std::string>(numerical_columns.size()));
  for (uint32_t i = 0; i < rows.size(); i++) {
    for (uint32_t j = 0; j < numerical_columns.size(); j++) {
      int value = std::stoi(rows[i][numerical_columns[j]]);
      if (neighbours.find(rows[i][source_col_num]) != neighbours.end()) {
        for (const auto& neighbour : neighbours.at(rows[i][source_col_num])) {
          value +=
              std::stoi(rows[node_id_map.at(neighbour)][numerical_columns[j]]);
        }
        processed_numerical_columns[i][j] =
            std::to_string(static_cast<float>(value) /
                           (neighbours.at(rows[i][source_col_num]).size() + 1));
      } else {
        processed_numerical_columns[i][j] = std::to_string(value);
      }
    }
  }

  return processed_numerical_columns;
}

std::vector<uint32_t> GraphDatasetFactory::getRelationshipColumns(
    const std::vector<std::string>& columns,
    const dataset::ColumnNumberMap& column_number_map) {
  std::vector<uint32_t> relationship_col_nums(columns.size());
  for (const auto& column : columns) {
    relationship_col_nums.push_back(column_number_map.at(column));
  }
  return relationship_col_nums;
}

std::pair<std::unordered_map<std::string, std::vector<std::string>>,
          ColumnNumberMap>
GraphDatasetFactory::getGraphStructureInfo(
    const std::vector<std::vector<std::string>>& rows) {
  uint32_t source_col_num = _column_number_map.at(_config->_source);
  if (!_config->_adj_list) {
    std::vector<uint32_t> relationship_col_nums = getRelationshipColumns(
        *_config->_relationship_columns, _column_number_map);

    return createGraph(rows, relationship_col_nums, source_col_num);
  }
  std::vector<std::string> nodes;
  for (const auto& temp : *_config->_adj_list) {
    nodes.push_back(temp.first);
  }
  return {*_config->_adj_list, ColumnNumberMap(nodes)};
}

dataset::CsvRolledBatch GraphDatasetFactory::getFinalProcessedData(
    const std::vector<std::vector<std::string>>& rows,
    const std::vector<uint32_t>& numerical_columns,
    const ColumnNumberMap& node_id_map,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        neighbours) {
  uint32_t source_col_num = _column_number_map.at(_config->_source);
  auto values = processNumerical(rows, numerical_columns, neighbours,
                                 source_col_num, node_id_map);

  auto copied_rows = rows;

  for (uint32_t i = 0; i < rows.size(); i++) {
    copied_rows[i].insert(copied_rows[i].end(), values[i].begin(),
                          values[i].end());
  }

  dataset::CsvRolledBatch input(copied_rows);

  return input;
}

std::vector<std::vector<std::string>> GraphDatasetFactory::getRawData(
    dataset::DataSource& data_loader) {
  std::vector<std::string> full_data;

  while (auto data =
             data_loader.nextBatch(DEFAULT_INTERNAL_FEATURIZATION_BATCH_SIZE)) {
    full_data.insert(full_data.end(), data->begin(), data->end());
  }

  std::vector<std::vector<std::string>> rows(full_data.size());

  for (uint32_t i = 0; i < full_data.size(); i++) {
    auto temp =
        dataset::ProcessorUtils::parseCsvRow(full_data[i], _config->_delimeter);
    rows[i] = std::vector<std::string>(temp.begin(), temp.end());
  }

  return rows;
}

dataset::PreprocessedVectorsPtr
GraphDatasetFactory::makeNumericalProcessedVectors(
    const std::vector<std::vector<std::string>>& rows,
    const ColumnNumberMap& node_id_map,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        neighbours) {
  std::vector<uint32_t> numerical_columns;
  std::vector<dataset::BlockPtr> input_blocks;
  uint32_t original_num_cols = _column_number_map.numCols();
  for (const auto& [col_name, data_type] : _config->_data_types) {
    uint32_t col_num = _column_number_map.at(col_name);
    if (auto numerical = asNumerical(data_type)) {
      numerical_columns.push_back(col_num);
    }
  }

  input_blocks.push_back(dataset::DenseArrayBlock::make(
      original_num_cols, numerical_columns.size()));

  auto key_vocab = dataset::ThreadSafeVocabulary::make(
      /* vocab_size= */ 0, /* limit_vocab_size= */ false);
  auto label_block = dataset::StringLookupCategoricalBlock::make(
      _column_number_map.at(_config->_source), key_vocab);

  auto processor = dataset::TabularFeaturizer::make(
      /* input_blocks= */ std::move(input_blocks),
      /* label_blocks= */ {std::move(label_block)},
      /* has_header= */ false, /* delimiter= */ _config->_delimeter,
      /* parallel= */ true);

  auto final_data =
      getFinalProcessedData(rows, numerical_columns, node_id_map, neighbours);

  return makePreprocessedVectors(processor, *key_vocab, final_data);
}

dataset::PreprocessedVectorsPtr
GraphDatasetFactory::makeFeatureProcessedVectors(
    const std::vector<std::vector<std::string>>& rows) {
  std::vector<dataset::BlockPtr> input_blocks;
  std::vector<dataset::TabularColumn> tabular_columns;

  for (const auto& [col_name, data_type] : _config->_data_types) {
    uint32_t col_num = _column_number_map.at(col_name);
    if ((col_name != _config->_target) && (col_name != _config->_source)) {
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
            (text_meta->average_n_words && text_meta->average_n_words <= 15)) {
          // text hash range of MAXINT is fine since features are later
          // hashed into a range. In fact it may reduce hash collisions.
          input_blocks.push_back(dataset::PairGramTextBlock::make(
              col_num, /* dim= */ std::numeric_limits<uint32_t>::max()));
        } else if (text_meta->contextual_encoding ==
                   TextEncodingType::Bigrams) {
          input_blocks.push_back(dataset::NGramTextBlock::make(
              col_num, /* n= */ 2,
              /* dim= */ std::numeric_limits<uint32_t>::max()));
        } else {
          input_blocks.push_back(dataset::NGramTextBlock::make(
              col_num, /* n= */ 1,
              /* dim= */ std::numeric_limits<uint32_t>::max()));
        }
      }
    }
  }

  input_blocks.push_back(std::make_shared<dataset::TabularHashFeatures>(
      /* columns= */ tabular_columns,
      /* output_range= */ std::numeric_limits<uint32_t>::max(),
      /* with_pairgrams= */ false));
  auto key_vocab = dataset::ThreadSafeVocabulary::make(
      /* vocab_size= */ 0, /* limit_vocab_size= */ false);
  auto label_block = dataset::StringLookupCategoricalBlock::make(
      _column_number_map.at(_config->_source), key_vocab);

  auto processor = dataset::TabularFeaturizer::make(
      /* input_blocks= */ std::move(input_blocks),
      /* label_blocks= */ {std::move(label_block)},
      /* has_header= */ false, /* delimiter= */ _config->_delimeter,
      /* parallel= */ true, /* hash_range= */ 100000);

  dataset::CsvRolledBatch final_data(rows);
  return makePreprocessedVectors(processor, *key_vocab, final_data);
}

std::vector<dataset::BlockPtr> GraphDatasetFactory::buildInputBlocks(
    const GraphConfigPtr& config) {
  UDTConfig feature_config(
      /* data_types= */ config->_data_types,
      /* temporal_tracking_relationships= */ {},
      /* target= */ config->_target,
      /* n_target_classes= */ config->_n_target_classes);

  TemporalRelationships empty_temporal_relationships;

  PreprocessedVectorsMap empty_vectors_map;

  return FeatureComposer::makeNonTemporalFeatureBlocks(
      feature_config, empty_temporal_relationships, empty_vectors_map,
      /*text_pairgrams_word_limit=*/5,
      /*contextual_columns=*/true);
}

void GraphDatasetFactory::findAllNeighboursForNode(  // NOLINT
    uint32_t k, const std::string& node_id, std::vector<bool>& visited,
    std::unordered_set<std::string>& neighbours,
    const std::unordered_map<std::string, std::vector<std::string>>&
        adjacency_list,
    const ColumnNumberMap& node_id_map) {
  if (k == 0) {
    return;
  }
  visited[node_id_map.at(node_id)] = true;

  for (const auto& neighbour : adjacency_list.at(node_id)) {
    if (!visited[node_id_map.at(neighbour)]) {
      neighbours.insert(neighbour);
      findAllNeighboursForNode(k - 1, neighbour, visited, neighbours,
                               adjacency_list, node_id_map);
    }
  }
}

dataset::PreprocessedVectorsPtr GraphDatasetFactory::makePreprocessedVectors(
    const dataset::TabularFeaturizerPtr& processor,
    dataset::ThreadSafeVocabulary& key_vocab, dataset::CsvRolledBatch rows) {
  auto batches = processor->featurize(rows);

  std::unordered_map<std::string, BoltVector> preprocessed_vectors(rows.size());

  for (uint32_t vec = 0; vec < batches[0].size(); vec++) {
    auto id = batches[1][vec].active_neurons[0];
    auto key = key_vocab.getString(id);
    preprocessed_vectors[key] = std::move(batches[0][vec]);
  }

  return std::make_shared<dataset::PreprocessedVectors>(
      std::move(preprocessed_vectors), processor->getInputDim());
}

}  // namespace thirdai::automl::data