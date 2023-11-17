#include <gtest/gtest.h>
#include <dataset/src/mach/MachIndex.h>
#include <iostream>
#include <ostream>

namespace thirdai::dataset::tests {

TEST(MachIndexSerializationTests, RNGStatePersists) {
  auto index =
      mach::MachIndex::make(/* num_buckets= */ 1000, /* num_hashes= */ 8);
  for (uint32_t i = 0; i < 100; i++) {
    index->insertWithRandomHashes(/* entity= */ i);
  }
  index->save("random_mach_index.ser");
  auto deserialized_index = mach::MachIndex::load("random_mach_index.ser");
  for (uint32_t i = 100; i < 200; i++) {
    index->insertWithRandomHashes(/* entity= */ i);
    deserialized_index->insertWithRandomHashes(/* entity= */ i);
  }
  for (uint32_t i = 0; i < 100; i++) {
    ASSERT_EQ(index->getHashes(i), deserialized_index->getHashes(i));
  }
}

}  // namespace thirdai::dataset::tests