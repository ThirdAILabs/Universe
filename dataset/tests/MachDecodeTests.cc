#include <gtest/gtest.h>
#include <dataset/src/mach/MachIndex.h>
#include <unordered_map>

namespace thirdai::dataset::tests {

void testMachDecode(const BoltVector& output) {
  std::unordered_map<uint32_t, std::vector<uint32_t>> entity_to_hash = {
      {0, {0, 1}}, {1, {1, 2}}, {2, {2, 3}}, {3, {0, 2}}};

  mach::MachIndex index(entity_to_hash, /* num_buckets= */ 4,
                        /* num_hashes= */ 2);

  uint32_t top_k = 3;
  auto results = index.decode(output, /* top_k = */ top_k,
                              /* num_buckets_to_eval = */ 2);
  ASSERT_EQ(results.size(), top_k);

  ASSERT_EQ(results[0].first, 2);
  ASSERT_NEAR(results[0].second, 0.9 / 2, 0.0001);
  ASSERT_EQ(results[1].first, 3);
  ASSERT_NEAR(results[1].second, 0.7 / 2, 0.0001);
  ASSERT_EQ(results[2].first, 1);
  ASSERT_NEAR(results[2].second, 0.5 / 2, 0.0001);
}

TEST(MachDecodeTests, DenseDecode) {
  std::vector<float> output_activations = {0.2, 0.0, 0.5, 0.4};
  auto output = BoltVector::makeDenseVector(output_activations);

  testMachDecode(output);
}

TEST(MachDecodeTests, SparseDecode) {
  auto output = BoltVector::makeSparseVector({2, 0, 3}, {0.5, 0.2, 0.4});

  testMachDecode(output);
}

}  // namespace thirdai::dataset::tests