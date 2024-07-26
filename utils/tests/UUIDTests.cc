#include <gtest/gtest.h>
#include <utils/UUID.h>
#include <unistd.h>
#include <unordered_set>

namespace thirdai::utils::uuid::tests {

TEST(UUIDTests, SameSeedDifferentTime) {
  UUIDGenerator gen1(123);
  UUIDGenerator gen2(123);

  uint64_t id1 = gen1();
  usleep(2000);
  uint64_t id2 = gen2();

  ASSERT_NE(id1, id2);
}

TEST(UUIDTests, SameTimeDifferentSeed) {
  UUIDGenerator gen1;
  UUIDGenerator gen2;

  // Since the time is per millisecond most of these UUIDS should have the same
  // time component.
  for (size_t i = 0; i < 100; i++) {
    uint64_t id1 = gen1();
    uint64_t id2 = gen2();

    ASSERT_NE(id1, id2);
  }
}

TEST(UUIDTests, TimeAndRandomComponentsVary) {
  UUIDGenerator gen;

  std::vector<uint64_t> ids(20);
  std::generate(ids.begin(), ids.end(), [&gen]() {
    usleep(2000);
    return gen();
  });

  for (size_t i = 0; i < ids.size(); i++) {
    for (size_t j = 0; j < i; j++) {
      uint64_t id1 = ids[i];
      uint64_t id2 = ids[j];

      ASSERT_NE(id1 >> 32, id2 >> 32);
      ASSERT_NE(id1 << 32, id2 << 32);
    }
  }
}

TEST(UUIDTests, AvalancheTest) {
  UUIDGenerator gen;

  std::vector<uint64_t> ids(100000);
  std::generate(ids.begin(), ids.end(), gen);

  std::unordered_set<uint64_t> unique(ids.begin(), ids.end());

  ASSERT_EQ(ids.size(), unique.size());
}

}  // namespace thirdai::utils::uuid::tests