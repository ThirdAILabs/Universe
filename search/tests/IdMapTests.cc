#include <gtest/gtest.h>
#include <search/src/inverted_index/id_map/IdMap.h>
#include <search/src/inverted_index/id_map/InMemoryIdMap.h>
#include <search/src/inverted_index/id_map/OnDiskIdMap.h>
#include <search/tests/InvertedIndexTestUtils.h>
#include <filesystem>
#include <vector>

namespace thirdai::search::tests {

class OnDiskIndexTests : public ::testing::Test {
 public:
  OnDiskIndexTests() { _prefix = randomPath() + "_"; }

  void TearDown() final {
    for (const auto& db : _dbs_created) {
      std::filesystem::remove_all(db);
    }
  }

  std::string tmpDbName() {
    std::string name = _prefix + std::to_string(_dbs_created.size()) + ".db";
    _dbs_created.push_back(name);
    return name;
  }

 private:
  std::string _prefix;
  std::vector<std::string> _dbs_created;
};

void testIdMap(IdMap& map) {
  map.put(1, {10, 100, 1000});
  map.put(2, {20});
  map.put(3, {30, 300});
  map.put(22, {20, 200});
  map.put(33, {300, 3000});

  ASSERT_EQ(map.get(1), std::vector<uint64_t>({10, 100, 1000}));
  ASSERT_EQ(map.get(2), std::vector<uint64_t>({20}));
  ASSERT_EQ(map.get(3), std::vector<uint64_t>({30, 300}));

  ASSERT_EQ(map.deleteValue(20), std::vector<uint64_t>({2}));
  ASSERT_EQ(map.deleteValue(40), std::vector<uint64_t>({}));
  ASSERT_EQ(map.deleteValue(30), std::vector<uint64_t>({}));

  ASSERT_EQ(map.get(3), std::vector<uint64_t>({300}));

  ASSERT_EQ(map.deleteValue(3000), std::vector<uint64_t>({}));
  ASSERT_EQ(map.deleteValue(300), std::vector<uint64_t>({3, 33}));
}

TEST(IdMapTests, OnDisk) {
  auto path = randomPath();
  OnDiskIdMap map(path);
  testIdMap(map);
  std::filesystem::remove_all(path);
}

TEST(IdMapTests, InMemory) {
  InMemoryIdMap map;
  testIdMap(map);
}

}  // namespace thirdai::search::tests