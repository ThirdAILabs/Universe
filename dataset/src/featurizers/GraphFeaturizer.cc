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
  _tabular_featurizer.updateColumnNumbers(column_number_map);
  _source_col.updateColumnNumber(column_number_map);
}

std::vector<std::vector<BoltVector>> GraphFeaturizer::featurize(
    ColumnarInputBatch& input_batch) {
  std::vector<BoltVector> batch_token_inputs(input_batch.size());

  auto batches = _tabular_featurizer.featurize(input_batch);

#pragma omp parallel for default(none) shared(input_batch, batch_token_inputs)
  for (size_t index_in_batch = 0; index_in_batch < input_batch.size();
       ++index_in_batch) {
    auto& sample = input_batch.at(index_in_batch);
    batch_token_inputs[index_in_batch] = buildNeighbourVector(sample);
  }

  return {std::move(batches[0]), std::move(batch_token_inputs),
          std::move(batches[1])};
}

std::vector<std::vector<BoltVector>> GraphFeaturizer::featurize(
    const LineInputBatch& input_batch) {
  CsvBatchRef input_batch_ref(input_batch, _delimiter, _expected_num_cols);
  return featurize(input_batch_ref);
}

BoltVector GraphFeaturizer::buildNeighbourVector(ColumnarInputSample& sample) {
  if (_neighbours.empty()) {
    throw std::invalid_argument("Haven't updated the neighbours.");
  }

  auto node_value = std::string(sample.column(_source_col));
  std::vector<uint32_t> indices(_num_neighbours, 0);
  uint32_t i = 0;
  if (_neighbours.count(node_value)) {
    for (auto it = _neighbours[node_value].begin();
         it != _neighbours[node_value].end(); it++, i++) {
      if (i >= _num_neighbours) {
        break;
      }
      indices[i] = _node_id_to_num_map.at(*it);
    }
  }
  std::vector<float> values(_num_neighbours, 1.0);

  return BoltVector::makeSparseVector(indices, values);
}

}  // namespace thirdai::dataset