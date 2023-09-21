#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/blocks/Augmentation.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/BlockList.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/blocks/NoOpAugmentation.h>
#include <iterator>
#include <memory>

namespace thirdai::dataset {

/**
 * Each BlockList passed in corresponds to a vector that gets returned from a
 * call to featurize.
 */
class TabularFeaturizer : public Featurizer {
 public:
  explicit TabularFeaturizer(std::vector<BlockList> block_lists,
                             bool has_header = false, char delimiter = ',',
                             bool parallel = true)
      : TabularFeaturizer(std::move(block_lists),
                          std::make_shared<NoOpAugmentation>(), has_header,
                          delimiter, parallel) {}

  TabularFeaturizer(std::vector<BlockList> block_lists,
                    AugmentationPtr augmentation, bool has_header = false,
                    char delimiter = ',', bool parallel = true)
      : _expects_header(has_header),
        _delimiter(delimiter),
        _parallel(parallel),
        _num_cols_in_header(std::nullopt),
        _block_lists(std::move(block_lists)),
        _augmentation(std::move(augmentation)),
        _expected_num_cols(0) {
    for (const auto& block_list : _block_lists) {
      _expected_num_cols =
          std::max(_expected_num_cols, block_list.expectedNumColumns());
    }
  }

  std::vector<BoltVector> featurize(ColumnarInputSample& input_sample);

  std::vector<std::vector<BoltVector>> featurize(
      ColumnarInputBatch& input_batch);

  std::vector<std::vector<BoltVector>> featurize(
      const LineInputBatch& input_batch) final;

  MapInputBatch convertToMapInputBatch(const LineInputBatch& input_batch,
                                       std::string key, std::string column_name,
                                       std::string header) final;

  bool expectsHeader() const final { return _expects_header; }

  void processHeader(const std::string& header) final;

  std::vector<uint32_t> getDimensions() final {
    std::vector<uint32_t> dims;
    dims.reserve(_block_lists.size());
    for (const auto& block_list : _block_lists) {
      dims.push_back(block_list.featureDim());
    }
    return dims;
  }

  size_t getNumDatasets() final { return _block_lists.size(); }

  void setParallelism(bool parallel) { _parallel = parallel; }

  // TODO(Any): The next two explanation functions only will explain through
  // the first set of block lists

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
      std::vector<BlockList> block_lists, bool has_header = false,
      char delimiter = ',', bool parallel = true) {
    return std::make_shared<TabularFeaturizer>(std::move(block_lists),
                                               has_header, delimiter, parallel);
  }

  static std::shared_ptr<TabularFeaturizer> make(
      std::vector<BlockList> block_lists, AugmentationPtr augmentation,
      bool has_header = false, char delimiter = ',', bool parallel = true) {
    return std::make_shared<TabularFeaturizer>(std::move(block_lists),
                                               std::move(augmentation),
                                               has_header, delimiter, parallel);
  }

 private:
  std::vector<std::vector<BoltVector>> featurizeSampleInBatch(
      ColumnarInputSample& input_sample);

  std::vector<std::vector<BoltVector>> consolidate(
      std::vector<std::vector<std::vector<BoltVector>>>&& vectors);

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Featurizer>(this), _expects_header, _delimiter,
            _parallel, _num_cols_in_header, _expected_num_cols, _block_lists,
            _augmentation);
  }

  // Private constructor for cereal.
  TabularFeaturizer() {}

  bool _expects_header;
  char _delimiter;
  bool _parallel;
  std::optional<uint32_t> _num_cols_in_header;

  std::vector<BlockList> _block_lists;
  AugmentationPtr _augmentation;
  uint32_t _expected_num_cols;
};

using TabularFeaturizerPtr = std::shared_ptr<TabularFeaturizer>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::TabularFeaturizer)