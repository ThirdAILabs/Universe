#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include "ProcessorUtils.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/BatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <algorithm>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace thirdai::dataset {

class GenericBatchProcessor : public BatchProcessor<BoltBatch, BoltBatch> {
 public:
  GenericBatchProcessor(
      std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks, bool has_header = false,
      char delimiter = ',', bool parallel = true,
      /*
        If hash_range has a value, then features from different blocks
        will be aggregated by hashing them to the same range but with
        different hash salts. Otherwise, the features will be treated
        as sparse vectors, which are then concatenated.
      */
      std::optional<uint32_t> hash_range = std::nullopt)
      : _expects_header(has_header),
        _delimiter(delimiter),
        _parallel(parallel),
        _hash_range(hash_range),
        _expected_num_cols(0),
        _input_blocks_dense(
            std::all_of(input_blocks.begin(), input_blocks.end(),
                        [](const std::shared_ptr<Block>& block) {
                          return block->isDense();
                        })),
        _label_blocks_dense(
            std::all_of(label_blocks.begin(), label_blocks.end(),
                        [](const std::shared_ptr<Block>& block) {
                          return block->isDense();
                        })),
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
        _label_blocks(std::move(label_blocks)) {
    _expected_num_cols = computeExpectedNumColumns();  // NOLINT
  }

  void updateColumnNumbers(const ColumnNumberMap& column_number_map) {
    _input_blocks.updateColumnNumbers(column_number_map);
    _label_blocks.updateColumnNumbers(column_number_map);
    _expected_num_cols = computeExpectedNumColumns();
  };

  std::tuple<BoltBatch, BoltBatch> createBatch(BatchInputRef& input_batch) {
    std::vector<BoltVector> batch_inputs(input_batch.size());
    std::vector<BoltVector> batch_labels(input_batch.size());

    auto& first_sample = input_batch.sample(0);
    if (auto error = first_sample.assertValid(_expected_num_cols)) {
      std::rethrow_exception(error);
    }
    _input_blocks.prepareForBatch(first_sample);
    _label_blocks.prepareForBatch(first_sample);

    /*
      These variables keep track of the presence of an erroneous input line.
      We do this instead of throwing an error directly because throwing
      an error inside an OpenMP structured block has undefined behavior.
    */
    std::exception_ptr featurization_err;

#pragma omp parallel for default(none) shared( \
    input_batch, batch_inputs, batch_labels, featurization_err) if (_parallel)
    for (size_t i = 0; i < input_batch.size(); ++i) {
      auto& columnar_sample = input_batch.sample(i);
      if (auto error = columnar_sample.assertValid(_expected_num_cols)) {
#pragma omp critical
        featurization_err = error;
        continue;
      }

      if (auto err = makeInputVectorInPlace(columnar_sample, batch_inputs[i])) {
#pragma omp critical
        featurization_err = err;
      }

      if (auto err = makeLabelVectorInPlace(columnar_sample, batch_labels[i])) {
        featurization_err = err;
      }
    }
    if (featurization_err) {
      std::rethrow_exception(featurization_err);
    }
    return std::make_tuple(BoltBatch(std::move(batch_inputs)),
                           BoltBatch(std::move(batch_labels)));
  }

  std::tuple<BoltBatch, BoltBatch> createBatch(
      const LineInputBatch& input_batch) final {
    BatchCsvLineInputRef input_batch_ref(input_batch, _delimiter);
    return createBatch(input_batch_ref);
  }

  bool expectsHeader() const final { return _expects_header; }

  void processHeader(const std::string& header) final { (void)header; }

  uint32_t getInputDim() const {
    return _hash_range.value_or(_input_blocks.featureDim());
  }

  uint32_t getLabelDim() const { return _label_blocks.featureDim(); }

  void setParallelism(bool parallel) { _parallel = parallel; }

  BoltVector makeInputVector(SingleInputRef& sample) {
    BoltVector vector;
    if (auto err = makeInputVectorInPlace(sample, vector)) {
      std::rethrow_exception(err);
    }
    return vector;
  }

  IndexToSegmentFeatureMap getIndexToSegmentFeatureMap(SingleInputRef& input) {
    BoltVector vector;
    auto segmented_vector =
        makeSegmentedFeatureVector(_input_blocks_dense, _hash_range,
                                   /* store_segment_feature_map= */ true);

    if (auto err = _input_blocks.addVectorSegment(input, *segmented_vector)) {
      std::rethrow_exception(err);
    }
    return segmented_vector->getIndexToSegmentFeatureMap();
  }

  Explanation explainFeature(SingleInputRef& input,
                             const SegmentFeature& segment_feature) {
    std::shared_ptr<Block> relevant_block =
        _input_blocks[segment_feature.segment_idx];

    return relevant_block->explainIndex(segment_feature.feature_idx, input);
  }

  static std::shared_ptr<GenericBatchProcessor> make(
      std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks, bool has_header = false,
      char delimiter = ',', bool parallel = true,
      std::optional<uint32_t> hash_range = std::nullopt) {
    return std::make_shared<GenericBatchProcessor>(input_blocks, label_blocks,
                                                   has_header, delimiter,
                                                   parallel, hash_range);
  }

 private:
  uint32_t computeExpectedNumColumns() {
    uint32_t expected_num_cols = _input_blocks.expectedNumColumns();
    expected_num_cols =
        std::max(expected_num_cols, _label_blocks.expectedNumColumns());
    return expected_num_cols;
  }

  std::exception_ptr makeInputVectorInPlace(SingleInputRef& sample,
                                            BoltVector& vector) {
    return makeVectorInPlace(sample, vector, _input_blocks, _input_blocks_dense,
                             _hash_range);
  }

  std::exception_ptr makeLabelVectorInPlace(SingleInputRef& sample,
                                            BoltVector& vector) {
    // Never hash labels.
    return makeVectorInPlace(sample, vector, _label_blocks, _label_blocks_dense,
                             /* hash_range= */ std::nullopt);
  }

  static std::exception_ptr makeVectorInPlace(
      SingleInputRef& sample, BoltVector& vector, BlockList& blocks,
      bool blocks_dense, std::optional<uint32_t> hash_range) noexcept {
    auto segmented_vector =
        makeSegmentedFeatureVector(blocks_dense, hash_range,
                                   /* store_segment_feature_map= */ false);
    if (auto err = blocks.addVectorSegment(sample, *segmented_vector)) {
      return err;
    }
    vector = segmented_vector->toBoltVector();
    return nullptr;
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
            _delimiter, _parallel, _hash_range, _expected_num_cols,
            _input_blocks_dense, _label_blocks_dense, _input_blocks,
            _label_blocks);
  }

  // Private constructor for cereal.
  GenericBatchProcessor() {}

  bool _expects_header;
  char _delimiter;
  bool _parallel;
  std::optional<uint32_t> _hash_range;

  uint32_t _expected_num_cols;
  bool _input_blocks_dense;
  bool _label_blocks_dense;
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
};

using GenericBatchProcessorPtr = std::shared_ptr<GenericBatchProcessor>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::GenericBatchProcessor)