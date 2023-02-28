#include "TabularFeaturizer.h"
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

std::vector<std::vector<BoltVector>> TabularFeaturizer::featurize(
    ColumnarInputBatch& input_batch) {
  std::vector<std::vector<std::vector<std::shared_ptr<SegmentedFeatureVector>>>>
      vectors(input_batch.size());
  
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
    try {
      auto a =
          buildSFV(_input_blocks, input_batch.at(index_in_batch), _hash_range);
      auto b = buildSFV(_label_blocks, input_batch.at(index_in_batch),
                        // Never hash labels
                        /* hash_range= */ std::nullopt);
      vectors[index_in_batch] = {{std::move(a), std::move(b)}};
    
      for (auto& augmentation : _augmentations) {
        vectors[index_in_batch] = augmentation->augment(
            std::move(vectors[index_in_batch]), input_batch.at(index_in_batch));
      }
    
    } catch (const std::exception& e) {
#pragma omp critical
      std::cout << e.what() << std::endl;
      featurization_err = std::make_exception_ptr(e);
      continue;
    }
  }
  if (featurization_err) {
    std::rethrow_exception(featurization_err);
  }
  return consolidate(std::move(vectors));
}

std::vector<std::vector<BoltVector>> TabularFeaturizer::consolidate(
    std::vector<
        std::vector<std::vector<std::shared_ptr<SegmentedFeatureVector>>>>&&
        vectors) {
  std::vector<std::vector<BoltVector>> bolt_vectors;
  for (auto& input_sample_vectors : vectors) {
    for (auto& output_sample_vectors : input_sample_vectors) {
      std::vector<BoltVector> output_sample_bolt_vectors;
      output_sample_bolt_vectors.reserve(output_sample_vectors.size());
      for (auto& column : output_sample_vectors) {
        output_sample_bolt_vectors.push_back(column->toBoltVector());
      }
      bolt_vectors.push_back(std::move(output_sample_bolt_vectors));
    }
  }
  return bolt_vectors;
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

void TabularFeaturizer::processHeader(const std::string& header) {
  _num_cols_in_header = CsvSampleRef(header, _delimiter,
                                     /* expected_num_cols= */ std::nullopt)
                            .size();
  dataset::ColumnNumberMap column_number_map(header, _delimiter);
  _input_blocks.updateColumnNumbers(column_number_map);
  _label_blocks.updateColumnNumbers(column_number_map);
  _expected_num_cols = std::max(_input_blocks.expectedNumColumns(),
                                _label_blocks.expectedNumColumns());
}

BoltVector TabularFeaturizer::makeInputVector(ColumnarInputSample& sample) {
  return buildSFV(_input_blocks, sample, _hash_range)->toBoltVector();
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
  if (!_augmentations.empty()) {
    throw std::runtime_error(
        "TabularFeaturizer does not support RCA when there are augmentation "
        "steps.");
  }
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
  if (!_augmentations.empty()) {
    throw std::runtime_error(
        "TabularFeaturizer does not support RCA when there are augmentation "
        "steps.");
  }
  std::shared_ptr<Block> relevant_block =
      _input_blocks[segment_feature.segment_idx];

  return relevant_block->explainIndex(segment_feature.feature_idx, input);
}

std::shared_ptr<SegmentedFeatureVector> TabularFeaturizer::buildSFV(
    BlockList& blocks, ColumnarInputSample& sample,
    std::optional<uint32_t> hash_range) {
  auto segmented_vector =
      makeSegmentedFeatureVector(blocks.areDense(), hash_range,
                                 /* store_segment_feature_map= */ false);
  if (auto err = blocks.addVectorSegments(sample, *segmented_vector)) {
    std::rethrow_exception(err);
  }
  return segmented_vector;
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
  // if (blocks_dense) {
  //   return std::make_shared<SegmentedDenseFeatureVector>(
  //       store_segment_feature_map);
  // }
  (void)blocks_dense;
  return std::make_shared<SegmentedSparseFeatureVector>(
      store_segment_feature_map);
}

}  // namespace thirdai::dataset
