#include <gtest/gtest.h>
#include <dataset/src/mach/MachDecode.h>
#include <dataset/src/mach/MachIndex.h>

namespace thirdai::dataset::tests {

class MockMachIndex : public mach::MachIndex {
 public:
  MockMachIndex() {}

  std::vector<uint32_t> hashAndStoreEntity(const std::string& string) final {
    std::unordered_map<std::string, std::vector<uint32_t>> map = {
        {"0", {2, 3}},
        {"1", {1, 2}},
        {"2", {0, 1}},
        {"3", {0, 3}},
    };

    return map[string];
  }

  std::vector<std::string> entitiesByHash(uint32_t hash_val) const final {
    std::unordered_map<uint32_t, std::vector<std::string>> map = {
        {0, {"2", "3"}},
        {1, {"1", "2"}},
        {2, {"0", "1"}},
        {3, {"0", "3"}},
    };

    return map[hash_val];
  }
};

TEST(MachDecodeTest, TestTopKUnlimitedDecode) {
  std::vector<float> output_activations = {0.2, 0.0, 0.5, 0.4};
  auto output = BoltVector::makeDenseVector(output_activations);

  uint32_t num_results = 3;
  uint32_t top_k = 3;
  auto results =
      mach::topKUnlimitedDecode(output, std::make_shared<MockMachIndex>(),
                                /* min_num_eval_results = */ num_results,
                                /* top_k_per_eval_aggregation = */ top_k);
  ASSERT_EQ(results.size(), num_results);

  ASSERT_EQ(results[0].first, "0");
  ASSERT_NEAR(results[0].second, 0.9, 0.0001);
  ASSERT_EQ(results[1].first, "3");
  ASSERT_NEAR(results[1].second, 0.6, 0.0001);
  ASSERT_EQ(results[2].first, "1");
  ASSERT_NEAR(results[2].second, 0.5, 0.0001);
}

}  // namespace thirdai::dataset::tests