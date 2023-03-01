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

std::exception_ptr buildVector(SegmentedFeatureVectorPtr& vector,
                               BlockList& blocks, ColumnarInputSample& sample);

std::shared_ptr<SegmentedFeatureVector> makeSegmentedFeatureVector(
    bool blocks_dense, std::optional<uint32_t> hash_range,
    bool store_segment_feature_map);

std::vector<std::vector<BoltVector>> TabularFeaturizer::featurize(
    ColumnarInputBatch& input_batch) {
  std::vector<std::vector<VectorBuilderRow>> vector_builders(
      input_batch.size());

  for (BlockList& block_list : _block_lists) {
    block_list.prepareForBatch(input_batch);
  }
  _augmentations.prepareForBatch(input_batch);
  /*
    These variables keep track of the presence of an erroneous input line.
    We do this instead of throwing an error directly because throwing
    an error inside an OpenMP structured block has undefined behavior.
  */
  std::exception_ptr featurization_err;
#pragma omp parallel for default(none) \
    shared(input_batch, vector_builders, featurization_err) if (_parallel)
  for (size_t index_in_batch = 0; index_in_batch < input_batch.size();
       ++index_in_batch) {
    if (auto error = featurizeSampleInBatch(index_in_batch, input_batch,
                                            vector_builders)) {
#pragma omp critical
      featurization_err = error;
      continue;
    }
  }
  if (featurization_err) {
    std::rethrow_exception(featurization_err);
  }
  return consolidate(std::move(vector_builders));
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

std::vector<std::vector<BoltVector>> TabularFeaturizer::consolidate(
    std::vector<std::vector<VectorBuilderRow>>&& vector_builders) {
  uint32_t n_samples = 0;
  std::vector<uint32_t> offsets(vector_builders.size() + 1);
  offsets[0] = 0;

  for (uint32_t input_sample_id = 0; input_sample_id < vector_builders.size();
       input_sample_id++) {
    n_samples += vector_builders[input_sample_id].size();
    offsets[input_sample_id + 1] = n_samples;
  }

  std::vector<std::vector<BoltVector>> output_batch(_block_lists.size());

  for (auto& column : output_batch) {
    column.resize(n_samples);
  }

#pragma omp parallel for default(none) \
    shared(vector_builders, output_batch, offsets, _block_lists)
  for (uint32_t input_sample_id = 0; input_sample_id < vector_builders.size();
       input_sample_id++) {
    auto& sample_augmentations = vector_builders.at(input_sample_id);
    for (uint32_t augmentation_id = 0;
         augmentation_id < sample_augmentations.size(); augmentation_id++) {
      auto output_sample_id = offsets[input_sample_id] + augmentation_id;
      for (uint32_t column_id = 0; column_id < _block_lists.size();
           column_id++) {
        auto& column = output_batch.at(column_id);
        auto& output_vector = column.at(output_sample_id);
        auto& vector_builder =
            sample_augmentations.at(augmentation_id).at(column_id);
        output_vector = vector_builder->toBoltVector();
      }
    }
  }
  return output_batch;
}

BoltVector TabularFeaturizer::makeInputVector(ColumnarInputSample& sample) {
  if (!_augmentations.empty()) {
    throw std::runtime_error(
        "Tabular featurizer cannot make single input vector when it has "
        "augmentations.");
  }
  SegmentedFeatureVectorPtr vector;
  if (auto err = buildVector(vector, _block_lists.at(0), sample)) {
    std::rethrow_exception(err);
  }
  return vector->toBoltVector();
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
  auto segmented_vector = makeSegmentedFeatureVector(
      _block_lists.at(0).areDense(), _block_lists.at(0).hashRange(),
      /* store_segment_feature_map= */ true);

  if (auto err =
          _block_lists.at(0).addVectorSegments(input, *segmented_vector)) {
    std::rethrow_exception(err);
  }
  return segmented_vector->getIndexToSegmentFeatureMap();
}

Explanation TabularFeaturizer::explainFeature(
    ColumnarInputSample& input, const SegmentFeature& segment_feature) {
  std::shared_ptr<Block> relevant_block =
      _block_lists.at(0)[segment_feature.segment_idx];

  return relevant_block->explainIndex(segment_feature.feature_idx, input);
}

std::exception_ptr TabularFeaturizer::featurizeSampleInBatch(
    uint32_t index_in_batch, ColumnarInputBatch& input_batch,
    std::vector<std::vector<VectorBuilderRow>>& vector_builders) {
  /*
    Try-catch block is for capturing invalid argument exceptions from
    input_batch.at(). Since we don't know the concrete type of the object
    returned by input_batch.at(), we can't take it out of the scope of the
    block. Thus, buildVector() also needs to be in this try-catch block.
  */
  try {
    auto& sample = input_batch.at(index_in_batch);
    vector_builders.at(index_in_batch).resize(1);
    vector_builders.at(index_in_batch).front().resize(_block_lists.size());
    for (size_t block_list_id = 0; block_list_id < _block_lists.size();
         block_list_id++) {
      if (auto err = buildVector(
              vector_builders.at(index_in_batch).front().at(block_list_id),
              _block_lists.at(block_list_id), sample)) {
        return err;
      }
    }
    vector_builders.at(index_in_batch) = _augmentations.augment(
        std::move(vector_builders.at(index_in_batch)), sample);
  } catch (std::exception& error) {
    return std::make_exception_ptr(error);
  }

  return nullptr;
}

std::exception_ptr buildVector(SegmentedFeatureVectorPtr& vector,
                               BlockList& blocks, ColumnarInputSample& sample) {
  vector = makeSegmentedFeatureVector(/* blocks_dense = */ blocks.areDense(),
                                      /* hash_range = */ blocks.hashRange(),
                                      /* store_segment_feature_map= */ false);
  return blocks.addVectorSegments(sample, *vector);
}

std::shared_ptr<SegmentedFeatureVector> makeSegmentedFeatureVector(
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

void TabularFeaturizer::processHeader(const std::string& header) {
  _num_cols_in_header = CsvSampleRef(header, _delimiter,
                                     /* expected_num_cols= */ std::nullopt)
                            .size();
  dataset::ColumnNumberMap column_number_map(header, _delimiter);

  _expected_num_cols = 0;
  for (BlockList& block_list : _block_lists) {
    block_list.updateColumnNumbers(column_number_map);
    _expected_num_cols =
        std::max(_expected_num_cols, block_list.expectedNumColumns());
  }
  _augmentations.updateColumnNumbers(column_number_map);
  _expected_num_cols =
      std::max(_expected_num_cols, _augmentations.expectedNumColumns());
}

}  // namespace thirdai::dataset
