#include "TabularFeaturizer.h"
#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include "ProcessorUtils.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <algorithm>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace thirdai::dataset {

void TabularFeaturizer::updateColumnNumbers(
    const ColumnNumberMap& column_number_map) {
  _input_blocks.updateColumnNumbers(column_number_map);
  _label_blocks.updateColumnNumbers(column_number_map);
  _expected_num_cols = std::max(_input_blocks.expectedNumColumns(),
                                _label_blocks.expectedNumColumns());
}

std::vector<std::vector<BoltVector>> TabularFeaturizer::featurize(
    ColumnarInputBatch& input_batch) {
  std::vector<BoltVector> batch_inputs(input_batch.size());
  std::vector<BoltVector> batch_labels(input_batch.size());

  _input_blocks.prepareForBatch(input_batch);
  _label_blocks.prepareForBatch(input_batch);

  /*
    These variables keep track of the presence of an erroneous input line.
    We do this instead of throwing an error directly because throwing
    an error inside an OpenMP structured block has undefined behavior.
  */
  std::exception_ptr featurization_err;
#pragma omp parallel for default(none) shared( \
    input_batch, batch_inputs, batch_labels, featurization_err) if (_parallel)
  for (size_t index_in_batch = 0; index_in_batch < input_batch.size();
       ++index_in_batch) {
    if (auto error = featurizeSampleInBatch(index_in_batch, input_batch,
                                            batch_inputs, batch_labels)) {
#pragma omp critical
      featurization_err = error;
      continue;
    }
  }
  if (featurization_err) {
    std::rethrow_exception(featurization_err);
  }
  return {std::move(batch_inputs), std::move(batch_labels)};
}

std::vector<std::vector<BoltVector>> TabularFeaturizer::featurize(
    const LineInputBatch& input_batch) {
  // If there isn't a header, we are forced to assume that every row will
  // have exactly as many columns as expected. Otherwise, we can assume that
  // every row will have the same number of columns as the header
  uint32_t expected_num_cols_in_batch =
      _num_cols_in_header.value_or(_expected_num_cols);
  CsvBatchRef input_batch_ref(input_batch, _delimiter,
                              expected_num_cols_in_batch);
  return featurize(input_batch_ref);
}

BoltVector TabularFeaturizer::makeInputVector(ColumnarInputSample& sample) {
  BoltVector vector;
  if (auto err = buildVector(vector, _input_blocks, sample, _hash_range)) {
    std::rethrow_exception(err);
  }
  return vector;
}

/**
 * This function is used in RCA.
 * The Generic featurizer creates input vectors by dispatching an input
 * sample through featurization blocks and combining these features using a
 * SegmentedFeatureVector. This function identifies the blocks that are
 * responsible for each feature in an input vector and maps them back to the
 * features produced by the blocks before they are combined.
 */
IndexToSegmentFeatureMap TabularFeaturizer::getIndexToSegmentFeatureMap(
    ColumnarInputSample& input) {
  BoltVector vector;
  auto segmented_vector =
      makeSegmentedFeatureVector(_input_blocks.areDense(), _hash_range,
                                 /* store_segment_feature_map= */ true);

  if (auto err = _input_blocks.addVectorSegments(input, *segmented_vector)) {
    std::rethrow_exception(err);
  }
  return segmented_vector->getIndexToSegmentFeatureMap();
}

Explanation TabularFeaturizer::explainFeature(
    ColumnarInputSample& input, const SegmentFeature& segment_feature) {
  std::shared_ptr<Block> relevant_block =
      _input_blocks[segment_feature.segment_idx];

  return relevant_block->explainIndex(segment_feature.feature_idx, input);
}

std::exception_ptr TabularFeaturizer::featurizeSampleInBatch(
    uint32_t index_in_batch, ColumnarInputBatch& input_batch,
    std::vector<BoltVector>& batch_inputs,
    std::vector<BoltVector>& batch_labels) {
  /*
    Try-catch block is for capturing invalid argument exceptions from
    input_batch.at(). Since we don't know the concrete type of the object
    returned by input_batch.at(), we can't take it out of the scope of the
    block. Thus, buildVector() also needs to be in this try-catch block.
  */
  try {
    auto& sample = input_batch.at(index_in_batch);
    if (auto err = buildVector(batch_inputs[index_in_batch], _input_blocks,
                               sample, _hash_range)) {
      return err;
    }
    return buildVector(batch_labels[index_in_batch], _label_blocks, sample,
                       // Label is never hashed.
                       /* hash_range= */ std::nullopt);
  } catch (std::invalid_argument& error) {
    return std::make_exception_ptr(error);
  }
}

std::exception_ptr TabularFeaturizer::buildVector(
    BoltVector& vector, BlockList& blocks, ColumnarInputSample& sample,
    std::optional<uint32_t> hash_range) {
  auto segmented_vector =
      makeSegmentedFeatureVector(blocks.areDense(), hash_range,
                                 /* store_segment_feature_map= */ false);
  if (auto err = blocks.addVectorSegments(sample, *segmented_vector)) {
    return err;
  }
  vector = segmented_vector->toBoltVector();
  return nullptr;
}

std::shared_ptr<SegmentedFeatureVector>
TabularFeaturizer::makeSegmentedFeatureVector(
    bool blocks_dense, std::optional<uint32_t> hash_range,
    bool store_segment_feature_map) {
  if (hash_range) {
    return std::make_shared<HashedSegmentedFeatureVector>(
        *hash_range, store_segment_feature_map);
  }
  // Dense vector if all blocks produce dense features, sparse vector
  // otherwise.
  if (blocks_dense) {
    return std::make_shared<SegmentedDenseFeatureVector>(
        store_segment_feature_map);
  }
  return std::make_shared<SegmentedSparseFeatureVector>(
      store_segment_feature_map);
}

}  // namespace thirdai::dataset
