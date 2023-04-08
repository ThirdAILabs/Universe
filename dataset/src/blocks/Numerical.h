#include "BlockInterface.h"
#include <string>

namespace thirdai::dataset {

class NumericalBlock final : public Block {
 public:
  explicit NumericalBlock(ColumnIdentifier col) : _col(std::move(col)) {}

  static auto make(ColumnIdentifier col) {
    return std::make_shared<NumericalBlock>(std::move(col));
  }

  uint32_t featureDim() const final { return 1; }

  bool isDense() const final { return true; }

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input) final {
    (void)index_within_block;
    (void)input;
    return {_col, "numerical_value_" + std::string(input.column(_col))};
  }

 protected:
  void buildSegment(ColumnarInputSample& input,
                    SegmentedFeatureVector& vec) final {
    float value = std::stof(std::string(input.column(_col)));
    vec.addDenseFeatureToSegment(value);
  }

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final {
    return {&_col};
  }

 private:
  ColumnIdentifier _col;

  // Private constructor for cereal.
  NumericalBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _col);
  }
};

using NumericalBlockPtr = std::shared_ptr<NumericalBlock>;

}  // namespace thirdai::dataset