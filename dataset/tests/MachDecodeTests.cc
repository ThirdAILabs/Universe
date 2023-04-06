#include <gtest/gtest.h>
#include <dataset/src/mach/MachDecode.h>
#include <dataset/src/mach/MachIndex.h>

namespace thirdai::dataset::tests {

class MachDecodeTest : public testing::Test {
 public:
  void static verifyIndex(const mach::NumericCategoricalMachIndexPtr& index) {
    auto map = index->_hash_to_entity;
    ASSERT_EQ(map[0].size(), 3);
    ASSERT_EQ(map[0][0], "0");
    ASSERT_EQ(map[0][1], "1");
    ASSERT_EQ(map[0][2], "3");

    ASSERT_EQ(map[1].size(), 1);
    ASSERT_EQ(map[1][0], "3");

    ASSERT_EQ(map[2].size(), 2);
    ASSERT_EQ(map[2][0], "2");
    ASSERT_EQ(map[2][1], "2");

    ASSERT_EQ(map[3].size(), 2);
    ASSERT_EQ(map[3][0], "0");
    ASSERT_EQ(map[3][1], "1");
  }

  static mach::NumericCategoricalMachIndexPtr makeMachIndex() {
    auto index = mach::NumericCategoricalMachIndex::make(
        /* output_range = */ 4, /* num_hashes = */ 2, /* max_elements = */ 4);

    index->hashAndStoreEntity("0");  // hashes to 0, 3
    index->hashAndStoreEntity("1");  // hashes to 0, 3
    index->hashAndStoreEntity("2");  // hashes to 2, 2
    index->hashAndStoreEntity("3");  // hashes to 0, 1

    verifyIndex(index);

    return index;
  }
};

TEST(MachDecodeTest, TestTopKUnlimitedDecode) {
  std::vector<float> output_activations = {0.2, 0.0, 0.5, 0.4};
  auto output = BoltVector::makeDenseVector(output_activations);

  uint32_t num_results = 3;
  uint32_t top_k = 3;
  auto results =
      mach::topKUnlimitedDecode(output, MachDecodeTest::makeMachIndex(),
                                /* min_num_eval_results = */ num_results,
                                /* top_k_per_eval_aggregation = */ top_k);
  ASSERT_EQ(results.size(), num_results);

  ASSERT_EQ(results[0].first, "2");
  ASSERT_EQ(results[0].second, 1);
  ASSERT_EQ(results[1].first, "1");
  ASSERT_NEAR(results[1].second, 0.6, 0.0001);
  ASSERT_EQ(results[2].first, "0");
  ASSERT_NEAR(results[2].second, 0.6, 0.0001);
}

}  // namespace thirdai::dataset::tests