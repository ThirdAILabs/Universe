#include <gtest/gtest.h>
#include <archive/src/Archive.h>
#include <archive/src/ArchiveMap.h>
<<<<<<< HEAD
=======
#include <archive/src/StringCipher.h>
>>>>>>> 943361c25720e736bed9e6d22dbafa6fd6dcf771
#include <archive/tests/Utils.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
<<<<<<< HEAD
=======
#include <unordered_map>
>>>>>>> 943361c25720e736bed9e6d22dbafa6fd6dcf771
#include <unordered_set>

namespace thirdai::ar::tests {

std::tuple<std::shared_ptr<ArchiveMap>, ConstArchivePtr, ConstArchivePtr,
           ConstArchivePtr>
simpleMap() {
  auto map = ArchiveMap::make();

  auto a = u64(10);
  auto b = str("hello");
  auto c = vecU32({1, 2, 3});

<<<<<<< HEAD
  map->set("a", a);
  map->set("b", b);
  map->set("c", c);
=======
  map->set("apple", a);
  map->set("bagel", b);
  map->set("chart", c);
>>>>>>> 943361c25720e736bed9e6d22dbafa6fd6dcf771

  return {map, a, b, c};
}

TEST(ArchiveMapTests, MapAccessing) {
  auto [map, a, b, c] = simpleMap();

  ASSERT_EQ(map->size(), 3);

<<<<<<< HEAD
  ASSERT_TRUE(map->contains("a"));
  ASSERT_TRUE(map->contains("b"));
  ASSERT_TRUE(map->contains("c"));

  ASSERT_EQ(map->get("a"), a);
  ASSERT_EQ(map->get("b"), b);
  ASSERT_EQ(map->get("c"), c);

  ASSERT_EQ(map->getAs<std::string>("b"), "hello");
  CHECK_EXCEPTION(map->getAs<F32>("b"),
=======
  ASSERT_TRUE(map->contains("apple"));
  ASSERT_TRUE(map->contains("bagel"));
  ASSERT_TRUE(map->contains("chart"));

  ASSERT_EQ(map->get("apple"), a);
  ASSERT_EQ(map->get("bagel"), b);
  ASSERT_EQ(map->get("chart"), c);

  ASSERT_EQ(map->getAs<std::string>("bagel"), "hello");
  CHECK_EXCEPTION(map->getAs<F32>("bagel"),
>>>>>>> 943361c25720e736bed9e6d22dbafa6fd6dcf771
                  "Attempted to convert archive of type 'Value[std::string]' "
                  "to type 'Value[float]'.",
                  std::runtime_error)

  ASSERT_EQ(map->getOpt<U64>("x"), std::nullopt);
<<<<<<< HEAD
  ASSERT_TRUE(map->getOpt<U64>("a").has_value());
  ASSERT_EQ(map->getOpt<U64>("a"), 10);
  CHECK_EXCEPTION(map->getOpt<F32>("b"),
=======
  ASSERT_TRUE(map->getOpt<U64>("apple").has_value());
  ASSERT_EQ(map->getOpt<U64>("apple"), 10);
  CHECK_EXCEPTION(map->getOpt<F32>("bagel"),
>>>>>>> 943361c25720e736bed9e6d22dbafa6fd6dcf771
                  "Attempted to convert archive of type 'Value[std::string]' "
                  "to type 'Value[float]'.",
                  std::runtime_error)

  ASSERT_EQ(map->getOr<U64>("x", 800), 800);
<<<<<<< HEAD
  ASSERT_EQ(map->getOr<U64>("a", 200), 10);
  CHECK_EXCEPTION(map->getOr<F32>("b", 4.4),
=======
  ASSERT_EQ(map->getOr<U64>("apple", 200), 10);
  CHECK_EXCEPTION(map->getOr<F32>("bagel", 4.4),
>>>>>>> 943361c25720e736bed9e6d22dbafa6fd6dcf771
                  "Attempted to convert archive of type 'Value[std::string]' "
                  "to type 'Value[float]'.",
                  std::runtime_error)

  CHECK_EXCEPTION(map->get("d"), "Map contains no value for key 'd'.",
                  std::out_of_range)
}

TEST(ArchiveMapTests, MapIterator) {
  auto [map, a, b, c] = simpleMap();

  std::unordered_map<std::string, ConstArchivePtr> visited;
  for (const auto& [k, v] : *map) {
    ASSERT_FALSE(visited.count(k));
    visited[k] = v;
  }

  ASSERT_EQ(visited.size(), 3);

  ASSERT_TRUE(visited.count("apple"));
  ASSERT_TRUE(visited.count("bagel"));
  ASSERT_TRUE(visited.count("chart"));

  ASSERT_EQ(visited.at("apple"), a);
  ASSERT_EQ(visited.at("bagel"), b);
  ASSERT_EQ(visited.at("chart"), c);
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

<<<<<<< HEAD
  ASSERT_EQ(loaded->getAs<U64>("a"), a->as<U64>());
  ASSERT_EQ(loaded->getAs<Str>("b"), b->as<Str>());
  ASSERT_EQ(loaded->getAs<VecU32>("c"), c->as<VecU32>());

  std::unordered_set<ConstArchivePtr> visited;
  for (const auto& [k, v] : loaded->map()) {
    ASSERT_FALSE(visited.count(v));
    visited.insert(v);
  }

  ASSERT_EQ(visited.size(), 3);
=======
  ASSERT_EQ(loaded->getAs<U64>("apple"), a->as<U64>());
  ASSERT_EQ(loaded->getAs<Str>("bagel"), b->as<Str>());
  ASSERT_EQ(loaded->getAs<VecU32>("chart"), c->as<VecU32>());

  std::unordered_map<std::string, ConstArchivePtr> visited;
  for (const auto& [k, v] : loaded->map()) {
    ASSERT_FALSE(visited.count(k));
    visited[k] = v;
  }

  ASSERT_EQ(visited.size(), 3);

  ASSERT_TRUE(visited.count("apple"));
  ASSERT_TRUE(visited.count("bagel"));
  ASSERT_TRUE(visited.count("chart"));
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
>>>>>>> 943361c25720e736bed9e6d22dbafa6fd6dcf771
}

}  // namespace thirdai::ar::tests