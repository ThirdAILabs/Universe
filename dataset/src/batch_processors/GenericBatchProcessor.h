#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/BatchProcessor.h>
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

class GenericBatchProcessor : public BatchProcessor {
 public:
  GenericBatchProcessor(
      std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks, bool has_header = false,
      std::string delimiter = ",", bool parallel = true,
      /*
        If hash_range has a value, then features from different blocks
        will be aggregated by hashing them to the same range but with
        different hash salts. Otherwise, the features will be treated
        as sparse vectors, which are then concatenated.
      */
      std::optional<uint32_t> hash_range = std::nullopt)
      : _expects_header(has_header),
        _delimiter(std::move(delimiter)),
        _parallel(parallel),
        _hash_range(hash_range),
        _num_cols_in_header(std::nullopt),
        /**
         * Here we copy input_blocks and label_blocks because when we
         * accept a vector representation of a Python List created by
         * PyBind11, the vector does not persist beyond this function
         * call, which results in segfaults later down the line.
         * It is therefore safest to just copy these vectors.
         * Furthermore, these vectors are cheap to copy since they contain a
         * small number of elements and each element is a pointer.
         */
        _input_blocks(std::move(input_blocks)),
        _label_blocks(std::move(label_blocks)),
        _expected_num_cols(std::max(_input_blocks.expectedNumColumns(),
                                    _label_blocks.expectedNumColumns())) {
    CSV::verifyDelimiterIsValid(_delimiter);
  }

  void updateColumnNumbers(const ColumnNumberMap& column_number_map) {
    _input_blocks.updateColumnNumbers(column_number_map);
    _label_blocks.updateColumnNumbers(column_number_map);
    _expected_num_cols = std::max(_input_blocks.expectedNumColumns(),
                                  _label_blocks.expectedNumColumns());
  };

  std::vector<BoltBatch> createBatch(ColumnarInputBatch& input_batch) {
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
    return {BoltBatch(std::move(batch_inputs)),
            BoltBatch(std::move(batch_labels))};
  }

  std::vector<BoltBatch> createBatch(const LineInputBatch& input_batch) final {
    // If there isn't a header, we are forced to assume that every row will
    // have exactly as many columns as expected. Otherwise, we can assume that
    // every row will have the same number of columns as the header
    uint32_t expected_num_cols_in_batch =
        _num_cols_in_header.value_or(_expected_num_cols);
    CsvBatchRef input_batch_ref(input_batch, _delimiter,
                                expected_num_cols_in_batch);
    return createBatch(input_batch_ref);
  }

  bool expectsHeader() const final { return _expects_header; }

  void processHeader(const std::string& header) final {
    _num_cols_in_header = CsvSampleRef(header, _delimiter,
                                       /* expected_num_cols= */ std::nullopt)
                              .size();
  }

  uint32_t getInputDim() const {
    return _hash_range.value_or(_input_blocks.featureDim());
  }

  uint32_t getLabelDim() const { return _label_blocks.featureDim(); }

  std::vector<uint32_t> getDimensions() final {
    std::vector<uint32_t> dims = {getInputDim(), getLabelDim()};
    return dims;
  }

  void setParallelism(bool parallel) { _parallel = parallel; }

  BoltVector makeInputVector(ColumnarInputSample& sample) {
    auto segmented_vector =
        makeSegmentedFeatureVector(_input_blocks.areDense(), _hash_range,
                                   /* store_segment_feature_map= */ false);
    if (auto err = _input_blocks.addVectorSegment(sample, *segmented_vector)) {
      std::rethrow_exception(err);
    }
    return segmented_vector->toBoltVector();
  }

  BoltVector makeLabelVector(ColumnarInputSample& sample) {
    // Never hash labels
    auto segmented_vector =
        makeSegmentedFeatureVector(_input_blocks.areDense(),
                                   /* hash_range= */ std::nullopt,
                                   /* store_segment_feature_map= */ false);
    if (auto err = _label_blocks.addVectorSegment(sample, *segmented_vector)) {
      std::rethrow_exception(err);
    }
    return segmented_vector->toBoltVector();
  }

  IndexToSegmentFeatureMap getIndexToSegmentFeatureMap(
      ColumnarInputSample& input) {
    BoltVector vector;
    auto segmented_vector =
        makeSegmentedFeatureVector(_input_blocks.areDense(), _hash_range,
                                   /* store_segment_feature_map= */ true);

    if (auto err = _input_blocks.addVectorSegment(input, *segmented_vector)) {
      std::rethrow_exception(err);
    }
    return segmented_vector->getIndexToSegmentFeatureMap();
  }

  Explanation explainFeature(ColumnarInputSample& input,
                             const SegmentFeature& segment_feature) {
    std::shared_ptr<Block> relevant_block =
        _input_blocks[segment_feature.segment_idx];

    return relevant_block->explainIndex(segment_feature.feature_idx, input);
  }

  static std::shared_ptr<GenericBatchProcessor> make(
      std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks, bool has_header = false,
      std::string delimiter = ",", bool parallel = true,
      std::optional<uint32_t> hash_range = std::nullopt) {
    return std::make_shared<GenericBatchProcessor>(
        input_blocks, label_blocks, has_header, std::move(delimiter), parallel,
        hash_range);
  }

 private:
  std::exception_ptr featurizeSampleInBatch(
      uint32_t index_in_batch, ColumnarInputBatch& input_batch,
      std::vector<BoltVector>& batch_inputs,
      std::vector<BoltVector>& batch_labels) {
    try {
      auto& sample = input_batch.sample(index_in_batch);
      batch_inputs[index_in_batch] = makeInputVector(sample);
      batch_labels[index_in_batch] = makeLabelVector(sample);
      return nullptr;
    } catch (std::exception_ptr& error) {
      return error;
    }
  }

  static std::shared_ptr<SegmentedFeatureVector> makeSegmentedFeatureVector(
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

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<BatchProcessor>(this), _expects_header,
            _delimiter, _parallel, _hash_range, _num_cols_in_header,
            _expected_num_cols, _input_blocks, _label_blocks);
  }

  // Private constructor for cereal.
  GenericBatchProcessor() {}

  bool _expects_header;
  std::string _delimiter;
  bool _parallel;
  std::optional<uint32_t> _hash_range;
  std::optional<uint32_t> _num_cols_in_header;

  /**
   * We save a copy of these vectors instead of just references
   * because using references will cause errors when given Python
   * lists through PyBind11. This is because while the PyBind11 creates
   * an std::vector representation of a Python list when passing it to
   * a C++ function, the vector does not persist beyond the function
   * call, so future references to the vector will cause a segfault.
   * Furthermore, these vectors are cheap to copy since they contain a
   * small number of elements and each element is a pointer.
   */
  BlockList _input_blocks;
  BlockList _label_blocks;
  uint32_t _expected_num_cols;
};

using GenericBatchProcessorPtr = std::shared_ptr<GenericBatchProcessor>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::GenericBatchProcessor)