#include "GraphBlocks.h"
#include <exceptions/src/Exceptions.h>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <utility>

namespace thirdai::dataset {

Explanation NormalizedNeighborVectorsBlock::explainIndex(
    uint32_t index_within_block, ColumnarInputSample& input) {
  (void)input;
  (void)index_within_block;
  throw exceptions::NotImplemented(
      "Graph blocks do not yet support explanations");
}

std::exception_ptr NormalizedNeighborVectorsBlock::buildSegment(
    ColumnarInputSample& input, SegmentedFeatureVector& vec) {
  char* end;
  uint64_t node_id =
      std::strtoul(input.column(_node_id_col).data(), &end, /* base = */ 10);

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

Explanation NeighborTokensBlock::explainIndex(
    uint32_t index_within_block, ColumnarInputSample& input) {
  (void)input;
  (void)index_within_block;
  throw exceptions::NotImplemented(
      "Graph blocks do not yet support explanations");
}

std::exception_ptr NeighborTokensBlock::buildSegment(
    ColumnarInputSample& input, SegmentedFeatureVector& vec) {
  char* end;
  uint64_t node_id =
      std::strtoul(input.column(_node_id_col).data(), &end, /* base = */ 10);

  std::vector<uint64_t> neighbors = _graph_ptr->neighbors(node_id);

  for (uint64_t neighbor : neighbors) {
    vec.addSparseFeatureToSegment(neighbor, /* value = */ 1);
  }

  return nullptr;
}

}  // namespace thirdai::dataset