#include "DyadicFeaturizer.h"
#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/InputTypes.h>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace thirdai::dataset {
std::vector<std::vector<BoltVector>> DyadicFeaturizer::featurize(
    const std::vector<std::string>& rows) {
  uint32_t expected_num_cols_in_batch = _num_cols_in_header.value_or(2);
  CsvBatchRef input_batch_ref(rows, _delimiter, expected_num_cols_in_batch);
  return featurize(input_batch_ref);
}

std::vector<std::vector<BoltVector>> DyadicFeaturizer::featurize(
    ColumnarInputBatch& inputs_ref) {
  size_t batch_size = inputs_ref.size();
  std::vector<std::vector<BoltVector>> bolt_batch(batch_size);

  std::exception_ptr exception_pointer = nullptr;
  std::atomic<bool> exception_thrown(false);

  for (size_t index = 0; index < batch_size; index++) {
    try {
      dataset::ColumnarInputSample& column_sample = inputs_ref.at(index);
      std::string feature_string = column_sample.column(
          ColumnIdentifier(_text_column, _column_number_map->at(_text_column)));

      std::vector<uint32_t> features =
          convertStringToUInt32FeatureArray(feature_string);
      auto featurized_vector = featurizeSingle(features);

      if (_label_column != std::nullopt) {
        std::string label_string = column_sample.column(
            ColumnIdentifier(_label_column.value(),
                             _column_number_map->at(_label_column.value())));
        std::vector<uint32_t> labels =
            convertStringToUInt32LabelArray(label_string);
        std::vector<uint32_t> mach_labels;

        if (_mach_label_block != std::nullopt) {
          for (const auto x : labels) {
            std::vector<uint32_t> mach_label =
                _mach_label_block.value()->index()->getHashes(x);
            mach_labels.insert(mach_labels.end(), mach_label.begin(),
                               mach_label.end());
          }
          auto mach_vector = thirdai::BoltVector::makeSparseVector(
              mach_labels, std::vector<float>(mach_labels.size(), 1));
          featurized_vector.push_back(mach_vector);
        }

        auto label_vector = thirdai::BoltVector::makeSparseVector(
            labels, std::vector<float>(labels.size(), 1));
        featurized_vector.push_back(label_vector);
      }

      bolt_batch[index] = featurized_vector;
    } catch (...) {
      exception_pointer = std::current_exception();
      exception_thrown.store(true);
    }
  }

  if (exception_thrown) {
    std::rethrow_exception(exception_pointer);
  }

  std::vector<std::vector<BoltVector>> transposed_batch(getNumDatasets());

  // auto printSize = [](const std::vector<std::vector<BoltVector>>& v) {
  //   size_t rows = v.size();
  //   size_t columns = (rows > 0) ? v[0].size() : 0;
  //   std::cout << "rows: " << rows << " and columns: " << columns <<
  //   std::endl;
  // };
  // printSize(bolt_batch);

  // printSize(transposed_batch);

  for (size_t index = 0; index < getNumDatasets(); index++) {
    transposed_batch[index] = std::vector<BoltVector>(batch_size);
  }
  for (size_t row = 0; row < bolt_batch.size(); row++) {
    for (size_t column = 0; column < bolt_batch[0].size(); column++) {
      transposed_batch[column][row] = bolt_batch[row][column].copy();
    }
  }
  return transposed_batch;
}

std::vector<std::vector<BoltVector>> DyadicFeaturizer::featurize(
    const MapInputBatch& map_input_batch) {
  dataset::MapBatchRef inputs_ref(map_input_batch);
  auto x = featurize(inputs_ref);
  return x;
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
    int end_point = tokens.size();
    int start_point = std::max(0, end_point - static_cast<int>(1 << interval));

    std::vector<uint32_t> temp_tokens(tokens.begin() + start_point,
                                      tokens.end());

    featurized_sample.emplace_back(thirdai::BoltVector::makeSparseVector(
        temp_tokens, std::vector<float>(temp_tokens.size(), 1.0)));
  }
  return featurized_sample;
}

template void DyadicFeaturizer::serialize(cereal::BinaryInputArchive&);
template void DyadicFeaturizer::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void DyadicFeaturizer::serialize(Archive& archive) {
  archive(_expects_header, _n_intervals, _text_column, _output_interval_prefix,
          _delimiter, _label_column, _label_delimiter, _mach_label_block,
          _num_cols_in_header, _column_number_map);
}

}  // namespace thirdai::dataset