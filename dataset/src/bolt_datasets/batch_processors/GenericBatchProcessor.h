#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/bolt_datasets/BatchProcessor.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>

namespace thirdai::dataset {

class GenericBatchProcessor : public BatchProcessor<bolt::BoltBatch> {
 public:
  GenericBatchProcessor(std::vector<std::shared_ptr<Block>> input_blocks,
                        std::vector<std::shared_ptr<Block>> label_blocks,
                        bool has_header = false, char delimiter = ',')
      : _expects_header(has_header),
        _delimiter(delimiter),
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
        _label_blocks(std::move(label_blocks)) {}

  std::optional<BoltDataLabelPair<bolt::BoltBatch>> createBatch(
      const std::vector<std::string>& rows) final {
    std::vector<bolt::BoltVector> batch_inputs(rows.size());
    std::vector<bolt::BoltVector> batch_labels(rows.size());

    // #pragma omp parallel for default(none) shared(rows, batch_inputs,
    // batch_labels)
    for (size_t i = 0; i < rows.size(); ++i) {
      auto columns = parseCsvRow(rows[i]);
      batch_inputs[i] = makeVector(columns, _input_blocks, _input_blocks_dense);
      batch_labels[i] = makeVector(columns, _label_blocks, _label_blocks_dense);
    }
    return std::make_pair(bolt::BoltBatch(std::move(batch_inputs)),
                          bolt::BoltBatch(std::move(batch_labels)));
  }

  bool expectsHeader() const final { return _expects_header; }

  void processHeader(const std::string& header) final { (void)header; }

  uint32_t getInputDim() const { return sumBlockDims(_input_blocks); }

  uint32_t getLabelDim() const { return sumBlockDims(_label_blocks); }

 private:
  // TODO(Geordie): Change to return string_view. Haven't done this yet since
  // we'll then have to change
  std::vector<std::string_view> parseCsvRow(const std::string& row) const {
    std::vector<std::string_view> parsed;
    size_t start = 0;
    size_t end = 0;
    while (end != std::string::npos) {
      end = row.find(_delimiter, start);
      size_t len = end == std::string::npos ? row.size() - start : end - start;
      parsed.push_back(std::string_view(row.data() + start, len));
      start = end + 1;
    }
    return parsed;
  }

  /**
   * Encodes a sample as a BoltVector according to the given blocks.
   */
  static bolt::BoltVector makeVector(
      std::vector<std::string_view>& sample,
      std::vector<std::shared_ptr<Block>>& blocks, bool blocks_dense) {
    std::shared_ptr<SegmentedFeatureVector> vec_ptr;

    // Dense vector if all blocks produce dense features, sparse vector
    // otherwise.
    if (blocks_dense) {
      vec_ptr = std::make_shared<SegmentedDenseFeatureVector>();
    } else {
      vec_ptr = std::make_shared<SegmentedSparseFeatureVector>();
    }

    // Let each block encode the input sample and adds a new segment
    // containing this encoding to the vector.
    for (auto& block : blocks) {
      block->addVectorSegment(sample, *vec_ptr);
    }
    return vec_ptr->toBoltVector();
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
    archive(cereal::base_class<BatchProcessor>(this), _input_blocks_dense,
            _label_blocks_dense, _input_blocks, _label_blocks);
  }

  // Private constructor for cereal.
  GenericBatchProcessor() {}

  bool _expects_header;
  char _delimiter;

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

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::GenericBatchProcessor)