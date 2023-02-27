#include "GraphBlocks.h"
#include <exceptions/src/Exceptions.h>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <utility>

namespace thirdai::dataset {

uint64_t parseUint64(const std::string_view& view) {
  char* end;
  return std::strtoul(view.data(), &end, /* base = */ 10);
}

uint64_t parseUint64(const ColumnIdentifier& identifier,
                     ColumnarInputSample& input) {
  return parseUint64(input.column(identifier));
}

uint64_t parseFloat(const ColumnIdentifier& identifier,
                    ColumnarInputSample& input) {
  char* end;
  return std::strtof(input.column(identifier).data(), &end);
}

std::vector<uint64_t> parseUint64Array(const std::string& array_string,
                                       char delimiter) {
  std::vector<std::string_view> parsed_array =
      ProcessorUtils::parseCsvRow(array_string, delimiter);
  std::vector<uint64_t> uint64_array;
  uint64_array.reserve(parsed_array.size());
  for (const auto& uint64_str : parsed_array) {
    uint64_array.push_back(parseUint64(uint64_str));
  }
  return uint64_array;
}

Explanation NormalizedNeighborVectorsBlock::explainIndex(
    uint32_t index_within_block, ColumnarInputSample& input) {
  (void)input;
  (void)index_within_block;
  throw exceptions::NotImplemented(
      "Graph blocks do not yet support explanations");
}

std::exception_ptr NormalizedNeighborVectorsBlock::buildSegment(
    ColumnarInputSample& input, SegmentedFeatureVector& vec) {
  uint64_t node_id = parseUint64(_node_id_col, input);
  std::vector<float> sum_neighbor_features(featureDim(), 0);

  for (uint64_t neighbor_id : _graph_ptr->neighbors(node_id)) {
    const std::vector<float> neighbor_feature =
        _graph_ptr->featureVector(neighbor_id);
    for (uint64_t d = 0; d < featureDim(); d++) {
      sum_neighbor_features.at(d) += neighbor_feature.at(d);
    }
  }

  // This normalizes the feature vector by the L1 sum
  float vector_sum =
      std::reduce(sum_neighbor_features.begin(), sum_neighbor_features.end());
  if (vector_sum != 0) {
    for (uint64_t d = 0; d < featureDim(); d++) {
      vec.addDenseFeatureToSegment(sum_neighbor_features.at(d) / vector_sum);
    }
  }

  return nullptr;
}

Explanation NeighborTokensBlock::explainIndex(uint32_t index_within_block,
                                              ColumnarInputSample& input) {
  (void)input;
  (void)index_within_block;
  throw exceptions::NotImplemented(
      "Graph blocks do not yet support explanations");
}

std::exception_ptr NeighborTokensBlock::buildSegment(
    ColumnarInputSample& input, SegmentedFeatureVector& vec) {
  uint64_t node_id = parseUint64(_node_id_col, input);

  std::vector<uint64_t> neighbors = _graph_ptr->neighbors(node_id);

  for (uint64_t neighbor : neighbors) {
    vec.addSparseFeatureToSegment(neighbor, /* value = */ 1);
  }

  return nullptr;
}

Explanation GraphBuilderBlock::explainIndex(uint32_t index_within_block,
                                            ColumnarInputSample& input) {
  (void)input;
  (void)index_within_block;
  throw exceptions::NotImplemented(
      "Graph blocks do not yet support explanations");
}

std::exception_ptr GraphBuilderBlock::buildSegment(
    ColumnarInputSample& input, SegmentedFeatureVector& vec) {
  (void)vec;

  uint64_t node_id = parseUint64(_node_id_col, input);

  std::vector<float> dense_feature_vector;
  dense_feature_vector.reserve(_feature_cols.size());
  for (const auto& feature_col : _feature_cols) {
    dense_feature_vector.push_back(parseFloat(feature_col, input));
  }

  std::vector<uint64_t> neighbors = parseUint64Array(
      input.column(_neighbor_col).data(), /* delimiter = */ ' ');

  _graph_ptr->insertNode(node_id, dense_feature_vector, neighbors);

  return nullptr;
}

}  // namespace thirdai::dataset