#include "GraphBlocks.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <dataset/src/utils/CsvParser.h>
#include <exceptions/src/Exceptions.h>
#include <cstdlib>
#include <exception>
#include <limits>
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
  if (array_string.empty()) {
    return std::vector<uint64_t>();
  }
  std::vector<std::string_view> parsed_array =
      parsers::CSV::parseLine(array_string, delimiter);
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
    std::vector<float> neighbor_feature;
    try {
      neighbor_feature = _graph_ptr->featureVector(neighbor_id);
    } catch (const automl::data::GraphConstructionError& e) {
      return std::make_exception_ptr(e);
    }
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

template void NormalizedNeighborVectorsBlock::serialize(
    cereal::BinaryInputArchive&);
template void NormalizedNeighborVectorsBlock::serialize(
    cereal::BinaryOutputArchive&);

template <class Archive>
void NormalizedNeighborVectorsBlock::serialize(Archive& archive) {
  archive(cereal::base_class<Block>(this), _node_id_col, _graph_ptr);
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

  std::vector<uint64_t> neighbors;
  try {
    neighbors = _graph_ptr->neighbors(node_id);
  } catch (const automl::data::GraphConstructionError& e) {
    return std::make_exception_ptr(e);
  }

  // We need to do this because this block is used as input to the Embedding
  // node, which does not support empty vectors. This is equivalent to making
  // all nodes with no neighbors have a single unique neighbor with id
  // std::numeric_limits<uint32_t>::max() - 1.
  if (neighbors.empty()) {
    neighbors.push_back(std::numeric_limits<uint32_t>::max() - 1);
  }

  for (uint64_t neighbor : neighbors) {
    vec.addSparseFeatureToSegment(neighbor, /* value = */ 1);
  }

  return nullptr;
}

template void NeighborTokensBlock::serialize(cereal::BinaryInputArchive&);
template void NeighborTokensBlock::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void NeighborTokensBlock::serialize(Archive& archive) {
  archive(cereal::base_class<Block>(this), _node_id_col, _graph_ptr);
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
      std::string(input.column(_neighbor_col)), /* delimiter = */ ' ');

  _graph_ptr->insertNode(node_id, dense_feature_vector, neighbors);

  return nullptr;
}

template void GraphBuilderBlock::serialize(cereal::BinaryInputArchive&);
template void GraphBuilderBlock::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void GraphBuilderBlock::serialize(Archive& archive) {
  archive(cereal::base_class<Block>(this), _node_id_col, _neighbor_col,
          _feature_cols, _graph_ptr);
}

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::NormalizedNeighborVectorsBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::NeighborTokensBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::GraphBuilderBlock)