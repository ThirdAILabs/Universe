#include <gtest/gtest.h>
#include <dataset/src/blocks/Augmentation.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/blocks/RecurrenceAugmentation.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <iostream>
#include <memory>
#include <unordered_map>

namespace thirdai::dataset::tests {

static constexpr uint32_t MAX_RECURRENCE = 5;
static constexpr uint32_t VOCAB_SIZE = 4;
static constexpr uint32_t INPUT_VECTOR_IDX = 0;
static constexpr uint32_t LABEL_VECTOR_IDX = 1;

RecurrenceAugmentation recurrenceAugmentation() {
  return RecurrenceAugmentation(
      /* sequence_column= */ {"sequence"}, /* delimiter= */ ' ', MAX_RECURRENCE,
      VOCAB_SIZE, INPUT_VECTOR_IDX, LABEL_VECTOR_IDX);
}

auto segmentedSparseFeatureVector(uint32_t dim,
                                  const std::vector<uint32_t>& indices) {
  auto vector = std::make_shared<SegmentedSparseFeatureVector>();
  vector->addFeatureSegment(dim);
  for (auto index : indices) {
    vector->addSparseFeatureToSegment(index, /* value= */ 1.0);
  }
  return vector;
}

TEST(RecurrenceAugmentationTests, ColumnNumberHandling) {
  auto augmentation = recurrenceAugmentation();

  ColumnNumberMap one_column("sequence", /* delimiter= */ ',');
  augmentation.updateColumnNumbers(one_column);
  ASSERT_EQ(augmentation.expectedNumColumns(), 1);

  ColumnNumberMap three_columns("other,column,sequence", /* delimiter= */ ',');
  augmentation.updateColumnNumbers(three_columns);
  ASSERT_EQ(augmentation.expectedNumColumns(), 3);
}

TEST(RecurrenceAugmentationTests, IsDense) {
  auto augmentation = recurrenceAugmentation();
  ASSERT_FALSE(augmentation.isDense(INPUT_VECTOR_IDX));
  ASSERT_FALSE(augmentation.isDense(LABEL_VECTOR_IDX));
}

TEST(RecurrenceAugmentationTests, AugmentingSingleRowOfEmptyVectors) {
  auto augmentation = recurrenceAugmentation();

  Vectors vectors = {{segmentedSparseFeatureVector(
                          /* dim= */ 0, /* indices= */ {}),
                      segmentedSparseFeatureVector(
                          /* dim= */ 0, /* indices= */ {})}};

  std::unordered_map<std::string, std::string> map_sample = {
      {"sequence", "a b c d"}};
  MapSampleRef sample(map_sample);
  vectors = augmentation.augment(std::move(vectors), sample);

  /*
    Expected augmentation result represents something like this
    (format is input_features,label_features):

    ,a_0
    a_0,b_1
    a_0 b_1,c_2
    a_0 b_1 c_2,d_3
    a_0 b_1 c_2 d_3,EOS_4

    We append EOS because the sequence is shorter than the maximum recurrence.

    Since this test processes the items in the sequence without parallelism, we
    can expect a, b, c, d, and EOS to get the IDs 0, 1, 2, 3, and 4
    respectively.

    Since we also encode the position, i.e. we encode a_0 instead of just a, we
    offset each ID by position * (vocab_size + 1); +1 is for EOS. For example,
    d_3 corresponds to a feature with index 3 * (4 + 1) + 3 = 18 and value 1.0.
  */
  std::vector<uint32_t> expected_indices = {
      0,   // a_0 = 0 * (4 + 1) + 0 = 0
      6,   // b_1 = 1 * (4 + 1) + 1 = 6
      12,  // c_2 = 2 * (4 + 1) + 2 = 12
      18,  // d_3 = 3 * (4 + 1) + 3 = 18
      24   // EOS_4 = 4 * (4 + 1) + 4 = 24
  };

  ASSERT_EQ(vectors.size(), expected_indices.size());
  for (uint32_t row = 0; row < vectors.size(); row++) {
    ASSERT_EQ(vectors[row].size(), 2);
    auto input_vec = vectors[row][0]->toBoltVector();
    auto label_vec = vectors[row][1]->toBoltVector();
    ASSERT_EQ(input_vec.len, row);
    ASSERT_EQ(label_vec.len, 1);
    for (uint32_t pos = 0; pos < row; pos++) {
      ASSERT_EQ(input_vec.active_neurons[pos], expected_indices[pos]);
      ASSERT_EQ(input_vec.activations[pos], 1.0);
    }
    ASSERT_EQ(label_vec.active_neurons[0], expected_indices[row]);
    ASSERT_EQ(label_vec.activations[0], 1.0);
  }
}

TEST(RecurrenceAugmentationTests, AugmentingManyRowsOfNonemptyVectors) {
  auto augmentation = recurrenceAugmentation();

  uint32_t initial_dim = 10;
  std::vector<std::vector<uint32_t>> initial_input_feats = {{1, 2}, {3, 4}};
  std::vector<std::vector<uint32_t>> initial_label_feats = {{5, 6}, {7, 8}};

  Vectors vectors = {
      // First row
      {segmentedSparseFeatureVector(
           /* dim= */ initial_dim, /* indices= */ initial_input_feats.at(0)),
       segmentedSparseFeatureVector(
           /* dim= */ initial_dim, /* indices= */ initial_label_feats.at(0))},
      // Second row
      {segmentedSparseFeatureVector(
           /* dim= */ initial_dim, /* indices= */ initial_input_feats.at(1)),
       segmentedSparseFeatureVector(
           /* dim= */ initial_dim, /* indices= */ initial_label_feats.at(1))}};
  uint32_t initial_n_feats = initial_input_feats.front().size();
  uint32_t initial_n_rows = vectors.size();

  std::unordered_map<std::string, std::string> map_sample = {
      {"sequence", "a b c d"}};
  MapSampleRef sample(map_sample);
  vectors = augmentation.augment(std::move(vectors), sample);

  // See comment in previous test to see why we expect these indices
  std::vector<uint32_t> expected_indices_without_offset = {0, 6, 12, 18, 24};
  uint32_t n_augmentations = expected_indices_without_offset.size();

  ASSERT_EQ(vectors.size(), n_augmentations * initial_n_rows);

  for (uint32_t input_row = 0; input_row < initial_n_rows; input_row++) {
    for (uint32_t augment_row = 0; augment_row < n_augmentations;
         augment_row++) {
      uint32_t row = input_row * n_augmentations + augment_row;
      ASSERT_EQ(vectors[row].size(), 2);
      auto input_vec = vectors[row][0]->toBoltVector();
      auto label_vec = vectors[row][1]->toBoltVector();
      ASSERT_EQ(input_vec.len, initial_n_feats + augment_row);
      ASSERT_EQ(label_vec.len, initial_n_feats + 1);

      for (uint32_t pos = 0; pos < initial_n_feats; pos++) {
        ASSERT_EQ(input_vec.active_neurons[pos],
                  initial_input_feats.at(input_row)[pos]);
        ASSERT_EQ(input_vec.activations[pos], 1.0);
        ASSERT_EQ(label_vec.active_neurons[pos],
                  initial_label_feats.at(input_row)[pos]);
        ASSERT_EQ(label_vec.activations[pos], 1.0);
      }

      for (uint32_t augment_feat_id = 0; augment_feat_id < augment_row;
           augment_feat_id++) {
        auto pos = initial_n_feats + augment_feat_id;
        ASSERT_EQ(
            input_vec.active_neurons[pos],
            expected_indices_without_offset[augment_feat_id] + initial_dim);
        ASSERT_EQ(input_vec.activations[pos], 1.0);
      }

      ASSERT_EQ(label_vec.active_neurons[initial_n_feats],
                expected_indices_without_offset[augment_row] + initial_dim);
      ASSERT_EQ(label_vec.activations[initial_n_feats], 1.0);
    }
  }
}

}  // namespace thirdai::dataset::tests