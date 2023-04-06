#include <gtest/gtest.h>
#include <dataset/src/mach/MachDecode.h>
#include <dataset/src/mach/MachIndex.h>

namespace thirdai::dataset::tests {

class MachDecodeTest : public testing::Test {
 public:
  static mach::NumericCategoricalMachIndexPtr makeMachIndex() {
    auto index = mach::NumericCategoricalMachIndex::make(
        /* output_range = */ 4, /* num_hashes = */ 2, /* max_elements = */ 4);

    index->_hash_to_entity[0] = {"0", "1"};
    index->_hash_to_entity[1] = {"0", "2", "3"};
    index->_hash_to_entity[2] = {"3"};
    index->_hash_to_entity[3] = {"1", "2"};

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

//   ASSERT_EQ(results[0].first, "1");
  ASSERT_EQ(results[0].second, 0.6);
  ASSERT_EQ(results[1].first, "3");
  ASSERT_EQ(results[1].second, 0.5);
  ASSERT_EQ(results[2].first, "2");
  ASSERT_EQ(results[2].second, 0.4);
}

}  // namespace thirdai::dataset::tests