#include <gtest/gtest.h>
#include <dataset/src/blocks/Augmentation.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace thirdai::dataset::tests {

class MockColumnarInputSample final : public ColumnarInputSample {
 public:
  explicit MockColumnarInputSample(
      std::unordered_map<std::string, std::string> columns)
      : _columns(std::move(columns)) {}

  std::string_view column(const ColumnIdentifier& column) final {
    return _columns.at(column.name());
  }

  uint32_t size() final { return _columns.size(); }

  std::unordered_map<std::string, std::string> _columns;
};

class MockAugmentation final : public Augmentation {
 public:
  MockAugmentation(ColumnIdentifier column, uint32_t expected_num_cols,
                   bool is_dense)
      : _column(std::move(column)),
        _expected_num_cols(expected_num_cols),
        _is_dense(is_dense) {}

  static auto make(ColumnIdentifier column, uint32_t expected_num_cols = 0,
                   bool is_dense = true) {
    return std::make_shared<MockAugmentation>(std::move(column),
                                              expected_num_cols, is_dense);
  }

  Vectors augment(Vectors&& vectors, ColumnarInputSample& input_sample) final {
    (void)input_sample;
    _vectors = vectors;
    _column_view = input_sample.column(_column);
    return vectors;
  }

  void updateColumnNumbers(const ColumnNumberMap& column_number_map) final {
    _column_number_map = column_number_map;
  }

  uint32_t expectedNumColumns() const final { return _expected_num_cols; }

  bool isDense(uint32_t vector_index) const final {
    (void)vector_index;
    return _is_dense;
  }

  ColumnIdentifier _column;
  uint32_t _expected_num_cols;
  bool _is_dense;
  Vectors _vectors;
  std::string_view _column_view;
  ColumnNumberMap _column_number_map;
};

Vectors makeVectors() {
  return {{std::make_shared<SegmentedSparseFeatureVector>(),
           std::make_shared<SegmentedSparseFeatureVector>()}};
}

TEST(AugmentationListTests, CorrectlyDispatchesAugment) {
  MockColumnarInputSample input_sample(
      {{"column_one", "column_one_value"}, {"column_two", "column_two_value"}});

  auto augmentation_one = MockAugmentation::make(/* column= */ {"column_one"});
  auto augmentation_two = MockAugmentation::make(/* column= */ {"column_two"});

  AugmentationList augmentations({augmentation_one, augmentation_two});

  auto vectors = makeVectors();

  augmentations.augment(std::move(vectors), input_sample);

  ASSERT_EQ(augmentation_one->_vectors, vectors);
  ASSERT_EQ(augmentation_two->_vectors, vectors);
  ASSERT_EQ(std::string(augmentation_one->_column_view), "column_one_value");
  ASSERT_EQ(std::string(augmentation_two->_column_view), "column_two_value");
}

TEST(AugmentationListTests, CorrectlyDispatchesUpdateColumnNumbers) {
  auto augmentation_one = MockAugmentation::make(/* column= */ {"column_one"});
  auto augmentation_two = MockAugmentation::make(/* column= */ {"column_two"});

  AugmentationList augmentations({augmentation_one, augmentation_two});

  ColumnNumberMap map("column_one,column_two", /* delimiter= */ ',');

  auto vectors = makeVectors();

  augmentations.updateColumnNumbers(map);

  ASSERT_EQ(augmentation_one->_column_number_map, map);
  ASSERT_EQ(augmentation_two->_column_number_map, map);
}

TEST(AugmentationListTests, CorrectlyDispatchesExpectedNumCols) {
  auto augmentation_one = MockAugmentation::make(/* column= */ {"column_one"},
                                                 /* expected_num_cols= */ 5);
  auto augmentation_two = MockAugmentation::make(/* column= */ {"column_two"},
                                                 /* expected_num_cols= */ 2);

  {
    AugmentationList augmentations({augmentation_one});
    ASSERT_EQ(augmentations.expectedNumColumns(), 5);
  }

  {
    AugmentationList augmentations({augmentation_two});
    ASSERT_EQ(augmentations.expectedNumColumns(), 2);
  }

  {
    // expectedNumColumns() is expected to be max of the expectedNumColumns of
    // the two augmentations.
    AugmentationList augmentations({augmentation_one, augmentation_two});
    ASSERT_EQ(augmentations.expectedNumColumns(), 5);
  }
}

TEST(AugmentationListTests, CorrectlyDispatchesExpectedNumCols) {
  auto augmentation_one =
      MockAugmentation::make(/* column= */ {"column_one"},
                             /* expected_num_cols= */ 0, /* is_dense= */ false);
  auto augmentation_two =
      MockAugmentation::make(/* column= */ {"column_two"},
                             /* expected_num_cols= */ 0, /* is_dense= */ true);

  {
    // No augmentation -> isDense defaults to true.
    AugmentationList augmentations({});
    ASSERT_EQ(augmentations.isDense(), true);
  }

  {
    AugmentationList augmentations({augmentation_one});
    ASSERT_EQ(augmentations.isDense(), false);
  }

  {
    AugmentationList augmentations({augmentation_two});
    ASSERT_EQ(augmentations.isDense(), true);
  }

  {
    // If any augmentation is not dense, then the list is not dense.
    AugmentationList augmentations({augmentation_one, augmentation_two});
    ASSERT_EQ(augmentations.isDense(), false);
  }
}

}  // namespace thirdai::dataset::tests