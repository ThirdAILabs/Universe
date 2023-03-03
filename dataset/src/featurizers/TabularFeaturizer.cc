#include "TabularFeaturizer.h"
#include "ProcessorUtils.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/utils/CsvParser.h>
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
  std::vector<std::vector<BoltVector>> featurized_batch(
      _block_lists.size(), std::vector<BoltVector>(input_batch.size()));

  /*
    Because throwing an error inside an OpenMP structured block has undefined
    behavior, we catch all errors inside the pragma omp parallel and then set
    this exception_ptr and rethrow after.
  */
  std::exception_ptr featurization_err;
#pragma omp parallel for default(none) \
    shared(input_batch, featurized_batch, featurization_err) if (_parallel)
  for (size_t index_in_batch = 0; index_in_batch < input_batch.size();
       ++index_in_batch) {
    try {
      featurizeSampleInBatch(index_in_batch, input_batch, featurized_batch);
    } catch (const std::exception& e) {
#pragma omp critical
      featurization_err = std::current_exception();
    }
  }
  if (featurization_err) {
    std::rethrow_exception(featurization_err);
  }
  return featurized_batch;
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
  return _block_lists.at(0).buildVector(sample)->toBoltVector();
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
  if (_block_lists.empty() || _block_lists.size() > 2) {
    throw std::runtime_error(
        "Explanations are not supported by this type of featurization.");
  }

  auto segmented_vector =
      _block_lists.at(0).buildVector(input,
                                     /* store_segment_feature_map= */ true);

  return segmented_vector->getIndexToSegmentFeatureMap();
}

Explanation TabularFeaturizer::explainFeature(
    ColumnarInputSample& input, const SegmentFeature& segment_feature) {
  if (_block_lists.empty() || _block_lists.size() > 2) {
    throw std::runtime_error(
        "Explanations are not supported by this type of featurization.");
  }

  std::shared_ptr<Block> relevant_block =
      _block_lists.at(0)[segment_feature.segment_idx];

  return relevant_block->explainIndex(segment_feature.feature_idx, input);
}

void TabularFeaturizer::featurizeSampleInBatch(
    uint32_t index_in_batch, ColumnarInputBatch& input_batch,
    std::vector<std::vector<BoltVector>>& featurized_batch) {
  auto& sample = input_batch.at(index_in_batch);
  for (size_t block_list_id = 0; block_list_id < _block_lists.size();
       block_list_id++) {
    featurized_batch.at(block_list_id).at(index_in_batch) =
        _block_lists.at(block_list_id).buildVector(sample)->toBoltVector();
  }
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
}

void TabularFeaturizer::processHeader(const std::string& header) {
  dataset::ColumnNumberMap column_number_map(header, _delimiter);
  _num_cols_in_header = column_number_map.size();

  _expected_num_cols = 0;
  _input_blocks.updateColumnNumbers(column_number_map);
  _expected_num_cols =
      std::max(_expected_num_cols, _input_blocks.expectedNumColumns());
  _label_blocks.updateColumnNumbers(column_number_map);
  _expected_num_cols =
      std::max(_expected_num_cols, _label_blocks.expectedNumColumns());
}

}  // namespace thirdai::dataset
