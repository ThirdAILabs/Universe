#pragma once

#include <cereal/access.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace thirdai::dataset {

class Augmentation {
 public:
  virtual void prepareForBatch(ColumnarInputBatch& incoming_batch) = 0;

  /**
   * Input: All segmented feature vectors (SFV) that correspond to a single
   * input sample.
   *
   * Output: Column-oriented matrix of BoltVectors, much like output of the
   * Featurizer::featurize method. The number of columns = number of SFVs in the
   * input. Number of rows = number of samples produced from a single output
   * sample. E.g.:
   *
   * Input: { InputSFV1 , InputSFV2 , LabelSFV }
   * Output:
   * {{ InputSFV1_Aug1 , InputSFV1_Aug2 , InputSFV1_Aug3 },
   *  { InputSFV2_Aug1 , InputSFV2_Aug2 , InputSFV2_Aug3 },
   *  { LabelSFV_Aug1  , LabelSFV_Aug2  , LabelSFV_Aug3  }}
   */
  virtual std::vector<std::vector<BoltVector>> augment(
      std::vector<SegmentedFeatureVectorPtr>&& builders,
      ColumnarInputSample& input_sample) = 0;

  virtual void updateColumnNumbers(
      const ColumnNumberMap& column_number_map) = 0;

  virtual ~Augmentation() = default;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

class PlaceholderBlock final : public Block {
 public:
  PlaceholderBlock(std::string name, uint32_t dim, bool dense,
                   ColumnIdentifier column_identifier)
      : _name(std::move(name)),
        _dim(dim),
        _dense(dense),
        _column_identifier(std::move(column_identifier)) {}

  uint32_t featureDim() const final { return _dim; }

  bool isDense() const final { return _dense; }

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input_row) final {
    (void)index_within_block;
    (void)input_row;
    throw std::runtime_error(_name + " currently does not support RCA.");
  }

  void buildSegment(ColumnarInputSample& input_row,
                    SegmentedFeatureVector& vec) final {
    (void)input_row;
    (void)vec;
  }

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final {
    return {&_column_identifier};
  }

 private:
  std::string _name;
  uint32_t _dim;
  bool _dense;
  ColumnIdentifier _column_identifier;

  PlaceholderBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _name, _dim, _dense,
            _column_identifier);
  }
};

using AugmentationPtr = std::shared_ptr<Augmentation>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::PlaceholderBlock)