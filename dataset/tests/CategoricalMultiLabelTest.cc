#include <gtest/gtest.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/encodings/categorical/CategoricalMultiLabel.h>

namespace thirdai::dataset {

TEST(CategoricalMultiLabelTest, TestLabelParsing) {
  auto multi_label_encoding =
      std::make_shared<CategoricalMultiLabel>(/* n_classes= */ 100);
  std::vector<std::shared_ptr<Block>> label_blocks = {
      std::make_shared<CategoricalBlock>(/* column_num= */ 0,
                                         multi_label_encoding),
      std::make_shared<CategoricalBlock>(/* column_num= */ 1,
                                         multi_label_encoding)};

  GenericBatchProcessor batch_processor(
      /* input_blocks= */ {}, /* label_blocks= */ label_blocks,
      /* has_header= */ false, /* delimiter= */ ' ');

  std::vector<std::string> rows = {"4,90,77 21,43,18,0", "55,67,82 49,2",
                                   "36 84,59,6"};

  auto batch = batch_processor.createBatch(rows);

  auto [data, labels] = std::move(batch);

  std::vector<std::vector<uint32_t>> expected_labels = {
      {4, 90, 77, 121, 143, 118, 100},
      {55, 67, 82, 149, 102},
      {36, 184, 159, 106}};

  EXPECT_EQ(labels.getBatchSize(), expected_labels.size());

  for (uint32_t vec_index = 0; vec_index < labels.getBatchSize(); vec_index++) {
    ASSERT_EQ(labels[vec_index].len, expected_labels.at(vec_index).size());
    for (uint32_t i = 0; i < labels[vec_index].len; i++) {
      ASSERT_EQ(labels[vec_index].active_neurons[i],
                expected_labels.at(vec_index).at(i));
      ASSERT_EQ(labels[vec_index].activations[i], 1.0);
    }
  }
}

}  // namespace thirdai::dataset
