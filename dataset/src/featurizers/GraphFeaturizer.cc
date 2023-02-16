#include "GraphFeaturizer.h"
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/featurizers/FeaturizerUtils.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <stdexcept>

namespace thirdai::dataset {

void GraphFeaturizer::processHeader(const std::string& header) {
  ColumnNumberMap column_number_map(header, _delimiter);
  _expected_num_cols = column_number_map.numCols();
  _input_blocks.updateColumnNumbers(column_number_map);
  _label_blocks.updateColumnNumbers(column_number_map);
  _source_col.updateColumnNumber(column_number_map);
}

std::vector<std::vector<BoltVector>> GraphFeaturizer::featurize(
    ColumnarInputBatch& input_batch) {
  std::vector<BoltVector> batch_inputs(input_batch.size());
  std::vector<BoltVector> batch_token_inputs(input_batch.size());
  std::vector<BoltVector> batch_labels(input_batch.size());

  _input_blocks.prepareForBatch(input_batch);
  _label_blocks.prepareForBatch(input_batch);

  std::exception_ptr featurization_err;
#pragma omp parallel for default(none)                                  \
    shared(input_batch, batch_inputs, batch_token_inputs, batch_labels, \
           featurization_err)
  for (size_t index_in_batch = 0; index_in_batch < input_batch.size();
       ++index_in_batch) {
    if (auto error =
            featurizeSampleInBatch(index_in_batch, input_batch, batch_inputs,
                                   batch_token_inputs, batch_labels)) {
#pragma omp critical
      featurization_err = error;
      continue;
    }
  }
  if (featurization_err) {
    std::rethrow_exception(featurization_err);
  }
  return {std::move(batch_inputs), std::move(batch_token_inputs),
          std::move(batch_labels)};
}

/*
The function updates the neighbours of each node.
*/
void GraphFeaturizer::updateNeighbours(const automl::Neighbours& neighbours) {
  for (auto [node, node_neighbours] : neighbours) {
    if (_neighbours.find(node) != _neighbours.end()) {
      for (const auto& neighbour : node_neighbours) {
        _neighbours[node].insert(neighbour);
      }
    } else {
      _neighbours[node] = node_neighbours;
    }
  }
}

/*
The purpose of the function is to assign a unique id for each node, written to
also support dynamic graph.
*/
void GraphFeaturizer::updateNodeIdMap(const ColumnNumberMap& node_id_map) {
  if (_node_id_to_num_map.empty()) {
    _node_id_to_num_map = node_id_map.getColumnNameToColNumMap();
    return;
  }
  auto temp = node_id_map.getColumnNumToColNameMap();
  for (const auto& node : temp) {
    auto present_size = _node_id_to_num_map.size();
    if (!_node_id_to_num_map.count(node)) {
      _node_id_to_num_map[node] = present_size;
    }
  }
}

std::vector<std::vector<BoltVector>> GraphFeaturizer::featurize(
    const LineInputBatch& input_batch) {
  CsvBatchRef input_batch_ref(input_batch, _delimiter, _expected_num_cols);
  return featurize(input_batch_ref);
}

std::exception_ptr GraphFeaturizer::featurizeSampleInBatch(
    uint32_t index_in_batch, ColumnarInputBatch& input_batch,
    std::vector<BoltVector>& batch_inputs,
    std::vector<BoltVector>& batch_token_inputs,
    std::vector<BoltVector>& batch_labels) {
  /*
    Try-catch block is for capturing invalid argument exceptions from
    input_batch.at(). Since we don't know the concrete type of the object
    returned by input_batch.at(), we can't take it out of the scope of the
    block. Thus, buildVector() also needs to be in this try-catch block.
  */
  try {
    auto& sample = input_batch.at(index_in_batch);
    if (auto err = FeaturizerUtils::buildVector(
            batch_inputs[index_in_batch], _input_blocks, sample, _hash_range)) {
      return err;
    }
    batch_token_inputs[index_in_batch] = buildNeighbourVector(sample);
    return FeaturizerUtils::buildVector(batch_labels[index_in_batch],
                                        _label_blocks, sample,
                                        // Label is never hashed.
                                        /* hash_range= */ std::nullopt);
  } catch (std::invalid_argument& error) {
    return std::make_exception_ptr(error);
  }
}

BoltVector GraphFeaturizer::buildNeighbourVector(ColumnarInputSample& sample) {
  if (_neighbours.empty()) {
    throw std::invalid_argument("Haven't updated the neighbours.");
  }

  auto node_value = std::string(sample.column(_source_col));
  std::vector<uint32_t> indices(_max_neighbors, 0);
  uint32_t i = 0;
  if (_neighbours.find(node_value) != _neighbours.end()) {
    for (auto it = _neighbours[node_value].begin();
         it != _neighbours[node_value].end(); it++, i++) {
      if (i >= _max_neighbors) {
        break;
      }
      indices[i] = _node_id_to_num_map.at(*it);
    }
  }
  std::vector<float> values(_max_neighbors, 1.0);

  return BoltVector::makeSparseVector(indices, values);
}

}  // namespace thirdai::dataset