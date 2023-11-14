#include <gtest/gtest.h>
#include <archive/src/Archive.h>
#include <archive/src/ArchiveMap.h>
#include <archive/src/StringCipher.h>
#include <archive/tests/Utils.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::ar::tests {

std::tuple<std::shared_ptr<ArchiveMap>, ConstArchivePtr, ConstArchivePtr,
           ConstArchivePtr>
simpleMap() {
  auto map = ArchiveMap::make();

  auto a = u64(10);
  auto b = str("hello");
  auto c = vecU32({1, 2, 3});

  map->set("apple", a);
  map->set("bagel", b);
  map->set("chart", c);

  return {map, a, b, c};
}

TEST(ArchiveMapTests, MapAccessing) {
  auto [map, a, b, c] = simpleMap();

  ASSERT_EQ(map->size(), 3);

  ASSERT_TRUE(map->contains("apple"));
  ASSERT_TRUE(map->contains("bagel"));
  ASSERT_TRUE(map->contains("chart"));

  ASSERT_EQ(map->get("apple"), a);
  ASSERT_EQ(map->get("bagel"), b);
  ASSERT_EQ(map->get("chart"), c);

  ASSERT_EQ(map->getAs<std::string>("bagel"), "hello");
  CHECK_EXCEPTION(map->getAs<F32>("bagel"),
                  "Attempted to convert archive of type 'Value[std::string]' "
                  "to type 'Value[float]'.",
                  std::runtime_error)

  ASSERT_EQ(map->getOpt<U64>("x"), std::nullopt);
  ASSERT_TRUE(map->getOpt<U64>("apple").has_value());
  ASSERT_EQ(map->getOpt<U64>("apple"), 10);
  CHECK_EXCEPTION(map->getOpt<F32>("bagel"),
                  "Attempted to convert archive of type 'Value[std::string]' "
                  "to type 'Value[float]'.",
                  std::runtime_error)

  ASSERT_EQ(map->getOr<U64>("x", 800), 800);
  ASSERT_EQ(map->getOr<U64>("apple", 200), 10);
  CHECK_EXCEPTION(map->getOr<F32>("bagel", 4.4),
                  "Attempted to convert archive of type 'Value[std::string]' "
                  "to type 'Value[float]'.",
                  std::runtime_error)

  CHECK_EXCEPTION(map->get("d"), "Map contains no value for key 'd'.",
                  std::out_of_range)
}

TEST(ArchiveMapTests, MapIterator) {
  auto [map, a, b, c] = simpleMap();

  std::unordered_set<ConstArchivePtr> visited;
  for (const auto& [k, v] : *map) {
    ASSERT_FALSE(visited.count(v));
    visited.insert(v);
  }

  ASSERT_EQ(visited.size(), 3);
  ASSERT_TRUE(visited.count(a));
  ASSERT_TRUE(visited.count(b));
  ASSERT_TRUE(visited.count(c));
}

TEST(ArchiveMapTests, Serialization) {
  auto [map, a, b, c] = simpleMap();

  std::stringstream buffer;
  serialize(map, buffer);
  auto loaded = deserialize(buffer);

  CHECK_EXCEPTION(loaded->list(),
                  "Expected to the archive to have type List but found 'Map'.",
                  std::runtime_error)

  ASSERT_EQ(loaded->map().size(), 3);

  ASSERT_EQ(loaded->getAs<U64>("apple"), a->as<U64>());
  ASSERT_EQ(loaded->getAs<Str>("bagel"), b->as<Str>());
  ASSERT_EQ(loaded->getAs<VecU32>("chart"), c->as<VecU32>());

  std::unordered_map<std::string, ConstArchivePtr> visited;
  for (const auto& [k, v] : loaded->map()) {
    ASSERT_FALSE(visited.count(k));
    visited[k] = v;
  }

  ASSERT_EQ(visited["apple"], a);
  ASSERT_EQ(visited["bagel"], b);
  ASSERT_EQ(visited["chart"], c);

  ASSERT_EQ(visited.size(), 3);
}

TEST(ArchiveMapTests, StringKeysAreHidden) {
  auto [map, a, b, c] = simpleMap();

  std::stringstream buffer;
  serialize(map, buffer);

  std::string serialized = buffer.str();

  ASSERT_EQ(serialized.find("apple"), std::string::npos);
  ASSERT_EQ(serialized.find("bagel"), std::string::npos);
  ASSERT_EQ(serialized.find("chart"), std::string::npos);

  ASSERT_NE(serialized.find(cipher("apple")), std::string::npos);
  ASSERT_NE(serialized.find(cipher("bagel")), std::string::npos);
  ASSERT_NE(serialized.find(cipher("chart")), std::string::npos);
}

}  // namespace thirdai::ar::tests