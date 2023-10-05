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
#include <vector>

namespace thirdai::dataset {

std::vector<BoltVector> TabularFeaturizer::featurize(
    ColumnarInputSample& input_sample) {
  std::vector<BoltVector> featurized_vectors(_block_lists.size());
  for (size_t i = 0; i < _block_lists.size(); i++) {
    featurized_vectors.at(i) =
        _block_lists.at(i).buildVector(input_sample)->toBoltVector();
  }
  return featurized_vectors;
}

std::vector<std::vector<BoltVector>> TabularFeaturizer::featurize(
    ColumnarInputBatch& input_batch) {
  if (input_batch.size() == 0) {
    throw std::invalid_argument("Cannot featurize empty batch.");
  }

  std::vector<std::vector<std::vector<BoltVector>>> featurized_batch(
      input_batch.size());

  for (BlockList& block_list : _block_lists) {
    block_list.prepareForBatch(input_batch);
  }

  /*
    Because throwing an error inside an OpenMP structured block has undefined
    behavior, we catch all errors inside the pragma omp parallel and then set
    this exception_ptr and rethrow after.
  */
  std::exception_ptr featurization_err;
#pragma omp parallel for default(none) \
    shared(input_batch, featurized_batch, featurization_err) if (_parallel)
  for (size_t sample_id = 0; sample_id < input_batch.size(); ++sample_id) {
    try {
      featurized_batch[sample_id] =
          featurizeSampleInBatch(input_batch.at(sample_id));
    } catch (const std::exception& e) {
#pragma omp critical
      featurization_err = std::current_exception();
    }
  }
  if (featurization_err) {
    std::rethrow_exception(featurization_err);
  }
  return consolidate(std::move(featurized_batch));
}

std::vector<std::vector<BoltVector>> TabularFeaturizer::featurize(
    const LineInputBatch& input_batch) {
  if (input_batch.empty()) {
    throw std::invalid_argument("Cannot featurize empty batch.");
  }

  // If there isn't a header, we are forced to assume that every row will
  // have exactly as many columns as expected. Otherwise, we can assume that
  // every row will have the same number of columns as the header
  uint32_t expected_num_cols_in_batch =
      _num_cols_in_header.value_or(_expected_num_cols);
  CsvBatchRef input_batch_ref(input_batch, _delimiter,
                              expected_num_cols_in_batch);
  return featurize(input_batch_ref);
}

MapInputBatch TabularFeaturizer::convertToMapInputBatch(
    const LineInputBatch& input_batch, const std::string& output_column_name,
    const std::string& input_column_name, const std::string& header) {
  dataset::ColumnNumberMap column_number_map(header, _delimiter);
  if (input_batch.empty()) {
    throw std::invalid_argument("Cannot featurize empty batch.");
  }

  uint32_t expected_num_cols_in_batch =
      _num_cols_in_header.value_or(_expected_num_cols);
  CsvBatchRef input_batch_ref(input_batch, _delimiter,
                              expected_num_cols_in_batch);
  MapInputBatch input_batches;

  for (size_t i = 0; i < input_batch_ref.size(); i++) {
    MapInput input;
    ColumnIdentifier col_id(column_number_map.at(input_column_name));

    input[output_column_name] = input_batch_ref.at(i).column(col_id);
    input_batches.push_back(input);
  }

  return input_batches;
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

std::vector<std::vector<BoltVector>> TabularFeaturizer::featurizeSampleInBatch(
    ColumnarInputSample& input_sample) {
  std::vector<SegmentedFeatureVectorPtr> builders(_block_lists.size());
  for (size_t block_list_id = 0; block_list_id < _block_lists.size();
       block_list_id++) {
    builders[block_list_id] =
        _block_lists.at(block_list_id).buildVector(input_sample);
  }

  return _augmentation->augment(std::move(builders), input_sample);
}

void TabularFeaturizer::processHeader(const std::string& header) {
  // TODO(Geordie): We don't need both num cols in header and expected num cols.
  dataset::ColumnNumberMap column_number_map(header, _delimiter);
  _num_cols_in_header = column_number_map.size();

  _expected_num_cols = 0;
  for (BlockList& block : _block_lists) {
    block.updateColumnNumbers(column_number_map);
    _expected_num_cols =
        std::max(_expected_num_cols, block.expectedNumColumns());
  }

  _augmentation->updateColumnNumbers(column_number_map);
}

std::vector<std::vector<BoltVector>> TabularFeaturizer::consolidate(
    std::vector<std::vector<std::vector<BoltVector>>>&& vectors) {
  uint32_t n_output_samples = 0;
  std::vector<uint32_t> offsets;
  offsets.reserve(vectors.size() + 1);
  offsets.push_back(0);
  for (auto& sample : vectors) {
    n_output_samples += sample.front().size();
    offsets.push_back(n_output_samples);
  }

  std::vector<std::vector<BoltVector>> outputs(
      _block_lists.size(), std::vector<BoltVector>(n_output_samples));

#pragma omp parallel for default(none) shared(vectors, outputs, offsets)
  for (uint32_t input_sample_id = 0; input_sample_id < vectors.size();
       input_sample_id++) {
    auto& augmented_vectors = vectors.at(input_sample_id);
    for (uint32_t column_id = 0; column_id < augmented_vectors.size();
         column_id++) {
      std::move(augmented_vectors[column_id].begin(),
                augmented_vectors[column_id].end(),
                outputs[column_id].begin() + offsets[input_sample_id]);
    }
  }

  return outputs;
}
}  // namespace thirdai::dataset
