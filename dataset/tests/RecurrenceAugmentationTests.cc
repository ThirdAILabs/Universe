#include <gtest/gtest.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/Augmentation.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/blocks/RecurrenceAugmentation.h>
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

auto buildersFromInitialFeatures(
    const std::vector<std::vector<uint32_t>>& initial_features,
    const std::vector<uint32_t>& initial_dims,
    RecurrenceAugmentation& augmentation) {
  assert(initial_features.size() == initial_dims.size());
  std::vector<SegmentedFeatureVectorPtr> builders(initial_dims.size());
  for (uint32_t builder_id = 0; builder_id < initial_dims.size();
       builder_id++) {
    uint32_t augmented_dim = 0;
    if (builder_id == INPUT_VECTOR_IDX) {
      augmented_dim = augmentation.inputBlock()->featureDim();
    } else if (builder_id == LABEL_VECTOR_IDX) {
      augmented_dim = augmentation.labelBlock()->featureDim();
    }
    builders[builder_id] = segmentedSparseFeatureVector(
        /* existing_dim= */ initial_dims[builder_id],
        /* augmented_dim= */ augmented_dim,
        /* indices= */ initial_features[builder_id]);
  }
  return builders;
}

void assertCorrectAugmentations(
    const std::vector<std::vector<uint32_t>>& initial_features,
    const std::vector<uint32_t>& initial_dims,
    RecurrenceAugmentation& augmentation) {
  auto builders =
      buildersFromInitialFeatures(initial_features, initial_dims, augmentation);
  uint32_t n_builders = builders.size();

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
  std::vector<uint32_t> expected_augmentation_feats = {
      1,   // a_0 = 0 * (4 + 1) + 1 = 1
      7,   // b_1 = 1 * (4 + 1) + 2 = 7
      13,  // c_2 = 2 * (4 + 1) + 3 = 13
      19,  // d_3 = 3 * (4 + 1) + 4 = 19
      20   // EOS_4 = 4 * (4 + 1) + 0 = 20
  };
  uint32_t n_augmentations = expected_augmentation_feats.size();

  // augmented is a n_builders by n_augmentations matrix of BoltVectors.
  ASSERT_EQ(augmented.size(), n_builders);
  ASSERT_EQ(augmented.front().size(), n_augmentations);

  for (uint32_t builder_id = 0; builder_id < initial_dims.size();
       builder_id++) {
    for (uint32_t aug_id = 0; aug_id < augmented.front().size(); aug_id++) {
      auto vector = augmented[builder_id][aug_id];

      // First assert that we have all the initial features.
      const auto& vec_initial_feats = initial_features[builder_id];
      for (uint32_t pos = 0; pos < vec_initial_feats.size(); pos++) {
        ASSERT_EQ(vector.active_neurons[pos], vec_initial_feats[pos]);
        ASSERT_EQ(vector.activations[pos], 1.0);
      }

      // Then assert that we have the augmentation features if relevant
      uint32_t initial_len = vec_initial_feats.size();
      uint32_t initial_dim = initial_dims[builder_id];

      switch (builder_id) {
        case INPUT_VECTOR_IDX:
          // Expect input vector to have the 0-th through aug_pos - 1-th
          // elements of expected_augmentation_feats.
          ASSERT_EQ(vector.len, vec_initial_feats.size() + aug_id);
          for (uint32_t aug_pos = 0; aug_pos < aug_id; aug_pos++) {
            ASSERT_EQ(vector.active_neurons[initial_len + aug_pos],
                      initial_dim + expected_augmentation_feats[aug_pos]);
            ASSERT_EQ(vector.activations[initial_len + aug_pos], 1.0);
          }
          break;

        case LABEL_VECTOR_IDX:
          // Expect label vector to have aug_pos-th element of
          // expected_augmentation_feats.
          ASSERT_EQ(vector.len, 1);
          ASSERT_EQ(vector.active_neurons[0],
                    expected_augmentation_feats[aug_id]);
          ASSERT_EQ(vector.activations[0], 1.0);
          break;

        default:
          // No augmentation feats
          ASSERT_EQ(vector.len, vec_initial_feats.size());
      }
    }
  }
}

TEST(RecurrenceAugmentationTests, IsDense) {
  auto augmentation = recurrenceAugmentation();
  ASSERT_FALSE(augmentation.inputBlock()->isDense());
  ASSERT_FALSE(augmentation.labelBlock()->isDense());
}

TEST(RecurrenceAugmentationTests, FeatureDim) {
  auto augmentation = recurrenceAugmentation();
  ASSERT_EQ(augmentation.inputBlock()->featureDim(), 25);
  ASSERT_EQ(augmentation.labelBlock()->featureDim(), 25);
}

TEST(RecurrenceAugmentationTests, AugmentingEmptyVectors) {
  auto augmentation = recurrenceAugmentation();
  assertCorrectAugmentations(/* initial_features= */ {{}, {}},
                             /* initial_dims= */ {0, 0}, augmentation);
}

/**
 * Tests that augmented input vectors retain features from before augmentation.
 */
TEST(RecurrenceAugmentationTests, AugmentingNonemptyVectors) {
  auto augmentation = recurrenceAugmentation();

  uint32_t initial_dim = 10;
  assertCorrectAugmentations(
      // Empty features at index 1 since it's for the label vector.
      /* initial_features= */ {{1, 2}, {}, {3, 4}},
      // Initial dim = 0 at index 1 since it's for the label vector.
      /* initial_dims= */ {10, 0, 10}, augmentation);
}

}  // namespace thirdai::dataset::tests