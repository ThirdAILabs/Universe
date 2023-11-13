#include <gtest/gtest.h>
#include <archive/src/Archive.h>
#include <archive/src/ArchiveMap.h>
#include <archive/tests/Utils.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace thirdai::ar::tests {

std::tuple<std::shared_ptr<ArchiveMap>, ConstArchivePtr, ConstArchivePtr,
           ConstArchivePtr>
simpleMap() {
  auto map = ArchiveMap::make();

  auto a = u64(10);
  auto b = str("hello");
  auto c = vecU32({1, 2, 3});

  map->set("a", a);
  map->set("b", b);
  map->set("c", c);

  return {map, a, b, c};
}

TEST(ArchiveMapTests, MapAccessing) {
  auto [map, a, b, c] = simpleMap();

  ASSERT_EQ(map->size(), 3);

  ASSERT_TRUE(map->contains("a"));
  ASSERT_TRUE(map->contains("b"));
  ASSERT_TRUE(map->contains("c"));

  ASSERT_EQ(map->get("a"), a);
  ASSERT_EQ(map->get("b"), b);
  ASSERT_EQ(map->get("c"), c);

  ASSERT_EQ(map->getAs<std::string>("b"), "hello");
  CHECK_EXCEPTION(map->getAs<F32>("b"),
                  "Attempted to convert archive of type 'Value[std::string]' "
                  "to type 'Value[float]'.",
                  std::runtime_error)

  ASSERT_EQ(map->getOpt<U64>("x"), std::nullopt);
  ASSERT_TRUE(map->getOpt<U64>("a").has_value());
  ASSERT_EQ(map->getOpt<U64>("a"), 10);
  CHECK_EXCEPTION(map->getOpt<F32>("b"),
                  "Attempted to convert archive of type 'Value[std::string]' "
                  "to type 'Value[float]'.",
                  std::runtime_error)

  ASSERT_EQ(map->getOr<U64>("x", 800), 800);
  ASSERT_EQ(map->getOr<U64>("a", 200), 10);
  CHECK_EXCEPTION(map->getOr<F32>("b", 4.4),
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

  ASSERT_EQ(loaded->getAs<U64>("a"), a->as<U64>());
  ASSERT_EQ(loaded->getAs<Str>("b"), b->as<Str>());
  ASSERT_EQ(loaded->getAs<VecU32>("c"), c->as<VecU32>());

  std::unordered_set<ConstArchivePtr> visited;
  for (const auto& [k, v] : loaded->map()) {
    ASSERT_FALSE(visited.count(v));
    visited.insert(v);
  }

  ASSERT_EQ(visited.size(), 3);
}

}  // namespace thirdai::ar::tests