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
    for (const auto& block : _input_blocks) {
      block->updateColumnNumbers(column_number_map);
    }
    for (const auto& block : _label_blocks) {
      block->updateColumnNumbers(column_number_map);
    }

    _expected_num_cols = computeExpectedNumColumns();
  };

  std::tuple<BoltBatch, BoltBatch> createBatch(
      const LineInputBatch& input_batch) final {
    std::vector<BoltVector> batch_inputs(input_batch.size());
    std::vector<BoltVector> batch_labels(input_batch.size());

    /*
      These variables keep track of the presence of an erroneous input line.
      We do this instead of throwing an error directly because throwing
      an error inside an OpenMP structured block has undefined behavior.
    */
    std::exception_ptr num_columns_error;
    std::exception_ptr block_err;

    prepareBlocksForNewBatch(input_batch.at(0));

#pragma omp parallel for default(none)                                 \
    shared(input_batch, batch_inputs, batch_labels, num_columns_error, \
           block_err) if (_parallel)
    for (size_t i = 0; i < input_batch.size(); ++i) {
      RowInput columns;
      if (auto err =
              makeRowInputFromLineInputInPlace(input_batch[i], columns)) {
#pragma omp critical
        num_columns_error = err;
        continue;
      }
      if (auto err = makeInputVectorInPlace(columns, batch_inputs[i])) {
#pragma omp critical
        block_err = err;
      }
      if (auto err = makeLabelVector(columns, batch_labels[i])) {
#pragma omp critical
        block_err = err;
      }
    }
    if (block_err) {
      std::rethrow_exception(block_err);
    }
    if (num_columns_error) {
      std::rethrow_exception(num_columns_error);
    }
    return std::make_tuple(BoltBatch(std::move(batch_inputs)),
                           BoltBatch(std::move(batch_labels)));
  }

  std::tuple<BoltBatch, BoltBatch> createBatch(
      const MapInputBatch& input_batch) {
    std::vector<BoltVector> batch_inputs(input_batch.size());
    std::vector<BoltVector> batch_labels(input_batch.size());

    /*
      These variables keep track of the presence of an erroneous input line.
      We do this instead of throwing an error directly because throwing
      an error inside an OpenMP structured block has undefined behavior.
    */
    std::exception_ptr num_columns_error;
    std::exception_ptr block_err;

    prepareBlocksForNewBatch(input_batch.at(0));

#pragma omp parallel for default(none)                                 \
    shared(input_batch, batch_inputs, batch_labels, num_columns_error, \
           block_err) if (_parallel)
    for (size_t i = 0; i < input_batch.size(); ++i) {
      if (auto err = makeInputVectorInPlace(input_batch[i], batch_inputs[i])) {
#pragma omp critical
        block_err = err;
      }
      if (auto err = makeLabelVector(input_batch[i], batch_labels[i])) {
#pragma omp critical
        block_err = err;
      }
    }
    if (block_err) {
      std::rethrow_exception(block_err);
    }
    if (num_columns_error) {
      std::rethrow_exception(num_columns_error);
    }
    return std::make_tuple(BoltBatch(std::move(batch_inputs)),
                           BoltBatch(std::move(batch_labels)));
  }

  bool expectsHeader() const final { return _expects_header; }

  void processHeader(const std::string& header) final { (void)header; }

  uint32_t getInputDim() const {
    return _hash_range.value_or(sumBlockDims(_input_blocks));
  }

  uint32_t getLabelDim() const { return sumBlockDims(_label_blocks); }

  void setParallelism(bool parallel) { _parallel = parallel; }

  template <typename InputType>
  void prepareBlocksForNewBatch(const InputType& sample) {
    for (auto& block : _input_blocks) {
      block->prepareForBatch(sample);
    }
    for (auto& block : _label_blocks) {
      block->prepareForBatch(sample);
    }
  }

  template <>
  void prepareBlocksForNewBatch(const LineInput& sample) {
    auto input_row = makeRowInputFromLineInput(sample);
    prepareBlocksForNewBatch(input_row);
  }

  template <typename InputType>
  BoltVector makeInputVector(const InputType& sample) {
    BoltVector vector;
    if (auto exception = makeInputVectorInPlace(sample, vector)) {
      std::rethrow_exception(exception);
    }
    return vector;
  }

  template <typename InputType>
  std::exception_ptr makeLabelVector(const InputType& sample,
                                     BoltVector& vector) {
    // Never hash labels.
    return makeVector(sample, vector, _label_blocks, _label_blocks_dense,
                      /* hash_range= */ std::nullopt);
  }

  template <typename InputType>
  IndexToSegmentFeatureMap getIndexToSegmentFeatureMap(const InputType& input) {
    BoltVector vector;
    auto segmented_vector =
        makeSegmentedFeatureVector(_input_blocks_dense, _hash_range,
                                   /* store_segment_feature_map= */ true);
    if (auto err = addFeaturesToSegmentedVector(input, *segmented_vector,
                                                _input_blocks)) {
      std::rethrow_exception(err);
    }
    return segmented_vector->getIndexToSegmentFeatureMap();
  }

  template <>
  IndexToSegmentFeatureMap getIndexToSegmentFeatureMap(const LineInput& input) {
    auto input_row = makeRowInputFromLineInput(input);
    return getIndexToSegmentFeatureMap(input_row);
  }

  template <typename InputType>
  Explanation explainFeature(const InputType& input,
                             const SegmentFeature& segment_feature) {
    std::shared_ptr<Block> relevant_block =
        _input_blocks[segment_feature.segment_idx];
    return relevant_block->explainIndex(segment_feature.feature_idx, input);
  }

  template <>
  Explanation explainFeature(const LineInput& input,
                             const SegmentFeature& segment_feature) {
    auto input_row = makeRowInputFromLineInput(input);
    return explainFeature(input_row, segment_feature);
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
    uint32_t expected_num_cols = maxNumColumnsExpectedByBlocks(_input_blocks);
    expected_num_cols = std::max(expected_num_cols,
                                 maxNumColumnsExpectedByBlocks(_label_blocks));
    return expected_num_cols;
  }

  static uint32_t maxNumColumnsExpectedByBlocks(
      const std::vector<BlockPtr>& blocks) {
    uint32_t expected_num_cols = 0;
    for (const auto& block : blocks) {
      try {
        expected_num_cols =
            std::max(block->expectedNumColumns(), expected_num_cols);

      } catch (const std::runtime_error& e) {
        (void)e;
      }
    }
    return expected_num_cols;
  }

  template <typename InputType>
  static std::exception_ptr makeVector(InputType& sample, BoltVector& vector,
                                       std::vector<BlockPtr>& blocks,
                                       bool blocks_dense,
                                       std::optional<uint32_t> hash_range) {
    auto segmented_vector =
        makeSegmentedFeatureVector(blocks_dense, hash_range,
                                   /* store_segment_feature_map= */ false);
    if (auto err =
            addFeaturesToSegmentedVector(sample, *segmented_vector, blocks)) {
      return err;
    }
    vector = segmented_vector->toBoltVector();
    return nullptr;
  }

  /**
   * Encodes a sample as a BoltVector according to the given blocks.
   */
  template <typename InputType>
  static std::exception_ptr addFeaturesToSegmentedVector(
      const InputType& sample, SegmentedFeatureVector& segmented_vector,
      std::vector<std::shared_ptr<Block>>& blocks) {
    for (auto& block : blocks) {
      if (auto err = block->addVectorSegment(sample, segmented_vector)) {
        return err;
      }
    }
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

  template <typename InputType>
  std::exception_ptr makeInputVectorInPlace(const InputType& sample,
                                            BoltVector& vector) {
    return makeVector(sample, vector, _input_blocks, _input_blocks_dense,
                      /* hash_range= */ _hash_range);
  }

  template <>
  std::exception_ptr makeInputVectorInPlace(const LineInput& sample,
                                            BoltVector& vector) {
    auto input_row = makeRowInputFromLineInput(sample);
    return makeVector(input_row, vector, _input_blocks, _input_blocks_dense,
                      /* hash_range= */ _hash_range);
  }

  std::exception_ptr makeRowInputFromLineInputInPlace(
      const LineInput& input, RowInput& input_row) const {
    input_row = ProcessorUtils::parseCsvRow(input, _delimiter);
    if (input_row.size() < _expected_num_cols) {
      std::stringstream error_ss;
      error_ss << "[ProcessorUtils::parseCsvRow] Expected "
               << _expected_num_cols << " columns delimited by '" << _delimiter
               << "' in each row of the dataset. Found row '" << input
               << "' with number of columns = " << input_row.size() << ".";
      return std::make_exception_ptr(std::invalid_argument(error_ss.str()));
    }
    return nullptr;
  }

  RowInput makeRowInputFromLineInput(const LineInput& input) {
    RowInput input_row;
    if (auto err = makeRowInputFromLineInputInPlace(input, input_row)) {
      std::rethrow_exception(err);
    }
    return input_row;
  }

  static uint32_t sumBlockDims(
      const std::vector<std::shared_ptr<Block>>& blocks) {
    uint32_t dim = 0;
    for (const auto& block : blocks) {
      dim += block->featureDim();
    }
    return dim;
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
  std::vector<std::shared_ptr<Block>> _input_blocks;
  std::vector<std::shared_ptr<Block>> _label_blocks;
};

using GenericBatchProcessorPtr = std::shared_ptr<GenericBatchProcessor>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::GenericBatchProcessor)