#include <gtest/gtest.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/Augmentation.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/blocks/RecurrenceAugmentation.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
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

auto segmentedSparseFeatureVector(uint32_t existing_dim, uint32_t augmented_dim,
                                  const std::vector<uint32_t>& indices) {
  auto vector = std::make_shared<SegmentedSparseFeatureVector>();
  vector->addFeatureSegment(existing_dim);
  for (auto index : indices) {
    vector->addSparseFeatureToSegment(index, /* value= */ 1.0);
  }
  vector->addFeatureSegment(augmented_dim);
  return vector;
}

TEST(RecurrenceAugmentationTests, IsDense) {
  auto augmentation = recurrenceAugmentation();
  ASSERT_FALSE(augmentation.inputBlock()->isDense());
  ASSERT_FALSE(augmentation.labelBlock()->isDense());
}

TEST(RecurrenceAugmentationTests, FeatureDim) {
  auto augmentation = recurrenceAugmentation();
  ASSERT_FALSE(augmentation.inputBlock()->isDense());
  ASSERT_FALSE(augmentation.labelBlock()->isDense());
}

TEST(RecurrenceAugmentationTests, AugmentingEmptyVectors) {
  auto augmentation = recurrenceAugmentation();

  std::vector<SegmentedFeatureVectorPtr> builders = {
      segmentedSparseFeatureVector(
          /* existing_dim= */ 0,
          /* augmented_dim= */ augmentation.inputBlock()->featureDim(),
          /* indices= */ {}),
      segmentedSparseFeatureVector(
          /* existing_dim= */ 0,
          /* augmented_dim= */ augmentation.labelBlock()->featureDim(),
          /* indices= */ {})};

  std::unordered_map<std::string, std::string> map_sample = {
      {"sequence", "a b c d"}};
  MapSampleRef sample(map_sample);
  auto augmented = augmentation.augment(std::move(builders), sample);

  /*
    Expected augmentation result represents something like this
    (format is input_features,label_features):

    ,a_0
    a_0,b_1
    a_0 b_1,c_2
    a_0 b_1 c_2,d_3
    a_0 b_1 c_2 d_3,EOS_4

    We append EOS because the sequence is shorter than the maximum recurrence.

    EOS will get ID 0. Since this test processes the items in the sequence
    without parallelism, we can expect a, b, c, and d to get the IDs 1, 2, 3,
    and 4 respectively.

    Since we also encode the position, i.e. we encode a_0 instead of just a, we
    offset each ID by position * (vocab_size + 1); +1 is for EOS. For example,
    d_3 corresponds to a feature with index 3 * (4 + 1) + 4 = 19 and value 1.0.
  */
  std::vector<uint32_t> expected_indices = {
      1,   // a_0 = 0 * (4 + 1) + 0 = 0
      7,   // b_1 = 1 * (4 + 1) + 1 = 6
      13,  // c_2 = 2 * (4 + 1) + 2 = 12
      19,  // d_3 = 3 * (4 + 1) + 3 = 18
      20   // EOS_4 = 4 * (4 + 1) + 4 = 24
  };

  ASSERT_EQ(augmented.size(), 2);
  ASSERT_EQ(augmented.front().size(), expected_indices.size());

  for (uint32_t row = 0; row < augmented.front().size(); row++) {
    auto input_vec = augmented[0][row];
    auto label_vec = augmented[1][row];
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

TEST(RecurrenceAugmentationTests, AugmentingNonemptyVectors) {
  auto augmentation = recurrenceAugmentation();

  uint32_t initial_dim = 10;
  std::vector<std::vector<uint32_t>> initial_input_feats = {{1, 2}, {3, 4}};

  std::vector<SegmentedFeatureVectorPtr> builders = {
      segmentedSparseFeatureVector(
          /* existing_dim= */ initial_dim,
          /* augmented_dim= */ augmentation.inputBlock()->featureDim(),
          /* indices= */ initial_input_feats.at(0)),
      segmentedSparseFeatureVector(
          /* existing_dim= */ 0,
          /* augmented_dim= */ augmentation.labelBlock()->featureDim(),
          /* indices= */ {}),
      segmentedSparseFeatureVector(
          /* existing_dim= */ initial_dim,
          /* augmented_dim= */ 0,
          /* indices= */ initial_input_feats.at(1))

  };
  uint32_t initial_n_feats = initial_input_feats.front().size();
  uint32_t n_builders = builders.size();

  std::unordered_map<std::string, std::string> map_sample = {
      {"sequence", "a b c d"}};
  MapSampleRef sample(map_sample);
  auto augmented = augmentation.augment(std::move(builders), sample);

  // See comment in previous test to see why we expect these indices
  std::vector<uint32_t> expected_element_ids = {1, 7, 13, 19, 20};
  uint32_t n_augmentations = expected_element_ids.size();

  ASSERT_EQ(augmented.size(), n_builders);
  ASSERT_EQ(augmented.front().size(), n_augmentations);

  for (uint32_t row = 0; row < augmented.front().size(); row++) {
    auto input_vec = augmented[0][row];
    auto other_input_vec = augmented[2][row];
    auto label_vec = augmented[1][row];
    ASSERT_EQ(input_vec.len, initial_n_feats + row);
    ASSERT_EQ(label_vec.len, 1);

    for (uint32_t pos = 0; pos < initial_n_feats; pos++) {
      ASSERT_EQ(input_vec.active_neurons[pos],
                initial_input_feats.at(0).at(pos));
      ASSERT_EQ(input_vec.activations[pos], 1.0);
      ASSERT_EQ(other_input_vec.active_neurons[pos],
                initial_input_feats.at(1).at(pos));
      ASSERT_EQ(other_input_vec.activations[pos], 1.0);
    }

    for (uint32_t pos = 0; pos < row; pos++) {
      ASSERT_EQ(input_vec.active_neurons[initial_n_feats + pos],
                initial_dim + expected_element_ids[pos]);
      ASSERT_EQ(input_vec.activations[initial_n_feats + pos], 1.0);
    }
    ASSERT_EQ(label_vec.active_neurons[0], expected_element_ids[row]);
    ASSERT_EQ(label_vec.activations[0], 1.0);
  }
}

}  // namespace thirdai::dataset::tests