#include "DyadicFeaturizer.h"
#include <bolt_vector/src/BoltVector.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <dataset/src/blocks/InputTypes.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace thirdai::dataset {
std::vector<std::vector<BoltVector>> DyadicFeaturizer::featurize(
    const std::vector<std::string>& rows) {
  uint32_t expected_num_cols_in_batch = _num_cols_in_header.value_or(2);
  CsvBatchRef input_batch_ref(rows, _delimiter, expected_num_cols_in_batch);

  size_t batch_size = input_batch_ref.size();
  std::vector<std::vector<BoltVector>> bolt_batch(batch_size);

  /**
   * For each ColumnarInputSample, convert the tokens to an array and generate
   * dyadic intervals for the array. Convert the intervals to a sparse bolt
   * vector and return [feature's bolt vector] + [label_vector]
   */
  std::exception_ptr exception_pointer = nullptr;
  std::atomic<bool> exception_thrown(false);

#pragma omp parallel for default(none)                                 \
    shared(bolt_batch, batch_size, input_batch_ref, exception_pointer, \
           exception_thrown)
  for (size_t index = 0; index < batch_size; index++) {
    try {
      dataset::ColumnarInputSample& column_sample = input_batch_ref.at(index);
      std::string feature_string =
          column_sample.column(_column_number_map->at(_text_column));
      std::string label_string =
          column_sample.column(_column_number_map->at(_label_column));

      std::vector<uint32_t> features =
          convertStringToUInt32FeatureArray(feature_string);
      auto featurized_vector = featurizeSingle(features);

      std::vector<uint32_t> labels =
          convertStringToUInt32LabelArray(label_string);
      auto label_vector = thirdai::BoltVector::makeSparseVector(
          labels, std::vector<float>(labels.size(), 1));
      featurized_vector.push_back(label_vector);
      bolt_batch[index] = featurized_vector;
    } catch (...) {
      exception_pointer = std::current_exception();
      exception_thrown.store(true);
    }
  }

  if (exception_thrown.load()) {
    try {
      std::rethrow_exception(exception_pointer);
    } catch (const std::exception& e) {
      std::cerr << "Caught exception: " << e.what() << std::endl;
    }
  }

  return bolt_batch;
}

std::vector<BoltVector> DyadicFeaturizer::featurizeSingle(
    const std::vector<uint32_t>& tokens) const {
  /**
   * This constructs dyadic features for the input token vector.
   * Note : Clips the latter part of the token vector if length is
   * greater than the context length.
   */
  std::vector<BoltVector> featurized_sample;
  for (size_t interval = 0; interval < _n_intervals; interval++) {
    int end_point = static_cast<int>(std::min(tokens.size(), _context_length));
    int start_point = std::max(0, end_point - static_cast<int>(1 << interval));

    std::vector<uint32_t> temp_tokens(tokens.begin() + start_point,
                                      tokens.begin() + end_point);

    featurized_sample.emplace_back(thirdai::BoltVector::makeSparseVector(
        temp_tokens, std::vector<float>(temp_tokens.size(), 1.0)));
  }
  return featurized_sample;
}

}  // namespace thirdai::dataset