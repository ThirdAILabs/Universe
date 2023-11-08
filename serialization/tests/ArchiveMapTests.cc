#include <gtest/gtest.h>
#include <serialization/src/Archive.h>
#include <serialization/src/ArchiveMap.h>
#include <serialization/tests/Utils.h>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace thirdai::ar::tests {

TEST(ArchiveMapTests, MapAccessing) {
  ArchiveMap map;

  auto a = u64(10);
  auto b = str("hello");
  auto c = vec(std::vector<uint32_t>{1, 2, 3});

  map.at("a") = a;
  map.at("b") = b;
  map.at("c") = c;

  ASSERT_EQ(map.size(), 3);

  ASSERT_TRUE(map.contains("a"));
  ASSERT_TRUE(map.contains("b"));
  ASSERT_TRUE(map.contains("c"));

  ASSERT_EQ(map.get("a"), a);
  ASSERT_EQ(map.get("b"), b);
  ASSERT_EQ(map.get("c"), c);

  CHECK_EXCEPTION(map.get("d"), "Map contains no value for key 'd'.",
                  std::invalid_argument);
}

TEST(ArchiveMapTests, MapIterator) {
  ArchiveMap map;

  auto a = u64(10);
  auto b = str("hello");
  auto c = vec(std::vector<uint32_t>{1, 2, 3});

  map.at("a") = a;
  map.at("b") = b;
  map.at("c") = c;

  std::unordered_set<ConstArchivePtr> visited;
  for (const auto& [k, v] : map) {
    ASSERT_FALSE(visited.count(v));
    visited.insert(v);
  }

  ASSERT_EQ(visited.size(), 3);
  ASSERT_TRUE(visited.count(a));
  ASSERT_TRUE(visited.count(b));
  ASSERT_TRUE(visited.count(c));
}

}  // namespace thirdai::ar::tests