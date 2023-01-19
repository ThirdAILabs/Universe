#pragma once

#include <_types/_uint32_t.h>
#include <dataset/src/BatchProcessor.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/utils/PreprocessedVectors.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <unordered_set>
namespace thirdai::dataset {

class GraphBatchProcessor : public BatchProcessor {
 public:
  GraphBatchProcessor(std::vector<std::shared_ptr<Block>> input_blocks,
                      std::vector<std::shared_ptr<Block>> label_blocks,
                      GenericBatchProcessorPtr preprocessed_processor,
                      std::vector<uint32_t> relationship_columns,
                      std::optional<std::vector<uint32_t>> numerical_columns,
                      const ThreadSafeVocabularyPtr& key_vocab,
                      uint32_t k_hop = 0, char delimeter = ',')
      : _input_blocks(std::move(input_blocks)),
        _label_blocks(std::move(label_blocks)),
        _preprocessed_processor(std::move(preprocessed_processor)),
        _relationship_columns(std::move(relationship_columns)),
        _numerical_columns(std::move(numerical_columns)),
        _vocab(*key_vocab),
        _k_hop(k_hop),
        _delimeter(delimeter) {
    _expected_num_cols = (std::max(_input_blocks.expectedNumColumns(),
                                   _label_blocks.expectedNumColumns()));
  }

  std::vector<BoltBatch> createBatch(const LineInputBatch& input_batch) final {
    CsvBatchRef input_batch_ref(input_batch, _delimeter, _expected_num_cols);

    createGraph(input_batch_ref);

    processNumerical(input_batch_ref, *_numerical_columns, _k_hop);

    createPreprocessedVectors(input_batch_ref);
    return {};
  }

  void createGraph(CsvBatchRef& input_batch_ref) {
    std::vector<std::vector<uint32_t>> adjacency_list_simulation(
        input_batch_ref.size());
#pragma omp parallel for default(none) \
    shared(input_batch_ref, adjacency_list_simulation) collapse(2)
    for (uint32_t i = 0; i < input_batch_ref.size(); i++) {
      for (uint32_t j = i + 1; j < input_batch_ref.size(); j++) {
        for (unsigned int relationship_col_num : _relationship_columns) {
          if (input_batch_ref.get(i).at(relationship_col_num) ==
              input_batch_ref.get(j).at(relationship_col_num)) {
            adjacency_list_simulation[i].push_back(j);
            adjacency_list_simulation[j].push_back(i);
            break;
          }
        }
      }
    }
    _adjacency_list = adjacency_list_simulation;
  }

  void processNumerical(CsvBatchRef& rows,
                        const std::vector<uint32_t>& numerical_columns,
                        uint32_t k_hop) {
    _neighbours.reserve(rows.size());
    std::vector<std::vector<std::string_view>> processed_numerical_columns(
        rows.size(), std::vector<std::string_view>(numerical_columns.size()));
#pragma omp parallel for default(none) shared( \
    rows, k_hop, numerical_columns, processed_numerical_columns) collapse(2)
    for (uint32_t i = 0; i < rows.size(); i++) {
      std::unordered_set<uint32_t> neighbours;
      std::vector<bool> visited(rows.size(), false);
      findAllNeighbours(k_hop, i, visited, neighbours);
      _neighbours[i] = neighbours;
      for (uint32_t j = 0; j < numerical_columns.size(); j++) {
        uint32_t value = 0;
        for (unsigned int neighbour : neighbours) {
          value +=
              stoi(std::string(rows.get(neighbour).at(numerical_columns[j])));
        }
        processed_numerical_columns[i][j] =
            std::to_string(value / neighbours.size());
      }
    }

    for (uint32_t i = 0; i < rows.size(); i++) {
      rows.insert(i, processed_numerical_columns[i]);
    }
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

  void createPreprocessedVectors(CsvBatchRef& rows) {
    auto prepocessed_batches = _preprocessed_processor->createBatch(rows);

    std::unordered_map<std::string, BoltVector> preprocessed_vectors(
        rows.size());

    for (uint32_t vec = 0; vec < prepocessed_batches[0].getBatchSize(); vec++) {
      auto id = prepocessed_batches[1][vec].active_neurons[0];
      auto key = _vocab.getString(id);
      preprocessed_vectors[key] = std::move(prepocessed_batches[0][vec]);
    }

    _vectors = std::make_shared<PreprocessedVectors>(
        std::move(preprocessed_vectors),
        _preprocessed_processor->getInputDim());
  }

 private:
  BlockList _input_blocks;
  BlockList _label_blocks;
  GenericBatchProcessorPtr _preprocessed_processor;
  std::vector<uint32_t> _relationship_columns;
  std::optional<std::vector<uint32_t>> _numerical_columns;
  ThreadSafeVocabulary _vocab;
  uint32_t _k_hop;
  char _delimeter;
  uint32_t _expected_num_cols;
  std::vector<std::vector<uint32_t>> _adjacency_list;
  std::vector<std::unordered_set<uint32_t>> _neighbours;
  PreprocessedVectorsPtr _vectors;
};

}  // namespace thirdai::dataset