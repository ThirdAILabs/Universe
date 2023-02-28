#include <gtest/gtest.h>
#include <dataset/src/blocks/Augmentations.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <memory>

namespace thirdai::dataset::tests {

TEST(AugmentationTests, SanityCheck) {
  std::cout << "A" << std::endl;
  auto feat = TabularFeaturizer::make(
      /* input_blocks= */ {}, /* label_blocks= */ {}, /* augmentations= */
      {std::make_shared<RecurrenceAugmentation>(/* sequence_column= */ 0,
                                                /* max_recurrence= */ 5,
                                                /* vocab_size= */ 5,
                                                /* input_vector_index= */ 0,
                                                /* label_vector_index= */ 1)});
  auto vecss = feat->featurize({"A B C D E", "B C D E A", "C D E A"});
  for (auto& vecs : vecss) {
    std::cout << "New sample" << std::endl;
    for (auto& vec : vecs) {
      std::cout << vec << std::endl;
    }
  }
}

}  // namespace thirdai::dataset::tests