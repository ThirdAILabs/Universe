#include <gtest/gtest.h>
#include <dataset/src/blocks/Augmentation.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

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
                   std::vector<bool> is_denses)
      : _column(std::move(column)),
        _expected_num_cols(expected_num_cols),
        _is_denses(std::move(is_denses)) {}

  static auto make(ColumnIdentifier column, uint32_t expected_num_cols = 0,
                   std::vector<bool> is_denses = {}) {
    return std::make_shared<MockAugmentation>(
        std::move(column), expected_num_cols, std::move(is_denses));
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
    return _is_denses.at(vector_index);
  }

  ColumnIdentifier _column;
  uint32_t _expected_num_cols;
  std::vector<bool> _is_denses;
  Vectors _vectors;
  std::string_view _column_view;
  ColumnNumberMap _column_number_map;
};

Vectors makeVectors() {
  auto vector_one = std::make_shared<SegmentedSparseFeatureVector>();
  vector_one->addFeatureSegment(10);
  vector_one->addSparseFeatureToSegment(/* index= */ 1, /* value= */ 1.0);

  auto vector_two = std::make_shared<SegmentedSparseFeatureVector>();
  vector_two->addFeatureSegment(10);
  vector_two->addSparseFeatureToSegment(/* index= */ 5, /* value= */ 1.0);

  return {{std::move(vector_one), std::move(vector_two)}};
}

void assertVectorsEqual(Vectors& lhs, Vectors& rhs) {
  ASSERT_EQ(lhs.size(), rhs.size());
  for (uint32_t i = 0; i < lhs.size(); i++) {
    ASSERT_EQ(lhs[i].size(), rhs[i].size());
    for (uint32_t j = 0; j < lhs[i].size(); j++) {
      auto lhs_bolt_vec = lhs[i][j]->toBoltVector();
      auto rhs_bolt_vec = rhs[i][j]->toBoltVector();
      ASSERT_EQ(lhs_bolt_vec.len, rhs_bolt_vec.len);
      for (uint32_t k = 0; k < lhs_bolt_vec.len; k++) {
        ASSERT_EQ(lhs_bolt_vec.active_neurons[k],
                  rhs_bolt_vec.active_neurons[k]);
        ASSERT_EQ(lhs_bolt_vec.activations[k], rhs_bolt_vec.activations[k]);
      }
    }
  }
}

TEST(AugmentationListTests, CorrectlyDispatchesAugment) {
  MockColumnarInputSample input_sample(
      {{"column_one", "column_one_value"}, {"column_two", "column_two_value"}});

  auto augmentation_one = MockAugmentation::make(/* column= */ {"column_one"});
  auto augmentation_two = MockAugmentation::make(/* column= */ {"column_two"});

  AugmentationList augmentations({augmentation_one, augmentation_two});

  auto vectors = makeVectors();
  auto moved_vectors = makeVectors();

  augmentations.augment(std::move(moved_vectors), input_sample);

  assertVectorsEqual(augmentation_one->_vectors, vectors);
  assertVectorsEqual(augmentation_two->_vectors, vectors);
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

  ASSERT_TRUE(augmentation_one->_column_number_map.equals(map));
  ASSERT_TRUE(augmentation_two->_column_number_map.equals(map));
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

TEST(AugmentationListTests, CorrectlyDispatchesIsDense) {
  auto augmentation_one = MockAugmentation::make(
      /* column= */ {"column_one"},
      /* expected_num_cols= */ 0, /* is_denses= */ {false, true});
  auto augmentation_two = MockAugmentation::make(
      /* column= */ {"column_two"},
      /* expected_num_cols= */ 0, /* is_denses= */ {true, true});

  {
    // No augmentation -> isDense defaults to true.
    AugmentationList augmentations({});
    ASSERT_EQ(augmentations.isDense(0), true);
    ASSERT_EQ(augmentations.isDense(1), true);
  }

  {
    AugmentationList augmentations({augmentation_one});
    ASSERT_EQ(augmentations.isDense(0), false);
    ASSERT_EQ(augmentations.isDense(1), true);
  }

  {
    AugmentationList augmentations({augmentation_two});
    ASSERT_EQ(augmentations.isDense(0), true);
    ASSERT_EQ(augmentations.isDense(1), true);
  }

  {
    AugmentationList augmentations({augmentation_one, augmentation_two});
    // If any augmentation is not dense, then the list is not dense.
    ASSERT_EQ(augmentations.isDense(0), false);
    ASSERT_EQ(augmentations.isDense(1), true);
  }
}

}  // namespace thirdai::dataset::tests