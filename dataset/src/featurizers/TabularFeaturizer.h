#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/blocks/BlockInterface.h>

namespace thirdai::dataset {

class TabularFeaturizer : public Featurizer {
 public:
  TabularFeaturizer(
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
                                    _label_blocks.expectedNumColumns())) {}

  void updateColumnNumbers(const ColumnNumberMap& column_number_map);

  std::vector<std::vector<BoltVector>> featurize(
      ColumnarInputBatch& input_batch);

  std::vector<std::vector<BoltVector>> featurize(
      const LineInputBatch& input_batch) final;

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

  size_t getNumDatasets() final { return 2; }

  void setParallelism(bool parallel) { _parallel = parallel; }

  BoltVector makeInputVector(ColumnarInputSample& sample);

  /**
   * This function is used in RCA.
   * The Generic featurizer creates input vectors by dispatching an input
   * sample through featurization blocks and combining these features using a
   * SegmentedFeatureVector. This function identifies the blocks that are
   * responsible for each feature in an input vector and maps them back to the
   * features produced by the blocks before they are combined.
   */
  IndexToSegmentFeatureMap getIndexToSegmentFeatureMap(
      ColumnarInputSample& input);

  Explanation explainFeature(ColumnarInputSample& input,
                             const SegmentFeature& segment_feature);

  static std::shared_ptr<TabularFeaturizer> make(
      std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks, bool has_header = false,
      char delimiter = ',', bool parallel = true,
      std::optional<uint32_t> hash_range = std::nullopt) {
    return std::make_shared<TabularFeaturizer>(
        std::move(input_blocks), std::move(label_blocks), has_header, delimiter,
        parallel, hash_range);
  }

 private:
  std::exception_ptr featurizeSampleInBatch(
      uint32_t index_in_batch, ColumnarInputBatch& input_batch,
      std::vector<BoltVector>& batch_inputs,
      std::vector<BoltVector>& batch_labels);

  static std::exception_ptr buildVector(BoltVector& vector, BlockList& blocks,
                                        ColumnarInputSample& sample,
                                        std::optional<uint32_t> hash_range);

  static std::shared_ptr<SegmentedFeatureVector> makeSegmentedFeatureVector(
      bool blocks_dense, std::optional<uint32_t> hash_range,
      bool store_segment_feature_map);

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Featurizer>(this), _expects_header, _delimiter,
            _parallel, _hash_range, _num_cols_in_header, _expected_num_cols,
            _input_blocks, _label_blocks);
  }

  // Private constructor for cereal.
  TabularFeaturizer() {}

  bool _expects_header;
  char _delimiter;
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

using TabularFeaturizerPtr = std::shared_ptr<TabularFeaturizer>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::TabularFeaturizer)