#include <gtest/gtest.h>
#include <serialization/src/Archive.h>
#include <serialization/src/ArchiveMap.h>
#include <serialization/tests/Utils.h>
#include <stdexcept>

namespace thirdai::ar::tests {

ArchiveMap simpleMap() {
  ArchiveMap map;

  map.at("a") = u64(10);
  map.at("b") = str("hello");
  map.at("c") = vec(std::vector<uint32_t>{1, 2, 3});

  return map;
}

TEST(ArchiveMapTests, MapAccessing) {
  ArchiveMap map = simpleMap();

  ASSERT_EQ(map.size(), 3);

  ASSERT_TRUE(map.contains("a"));
  ASSERT_TRUE(map.contains("b"));
  ASSERT_TRUE(map.contains("c"));

  ASSERT_EQ(map.at("a")->get<uint64_t>(), 10);
  ASSERT_EQ(map.at("b")->get<std::string>(), "hello");
  std::vector<uint32_t> list_val = {1, 2, 3};
  ASSERT_EQ(map.at("c")->get<std::vector<uint32_t>>(), list_val);

  CHECK_EXCEPTION(map.get("d"), "Map contains no value for key 'd'.",
                  std::invalid_argument);
}

TEST(ArchiveMapTests, MapIterator) {
  ArchiveMap map = simpleMap();

  size_t uint_cnt = 0;
  size_t str_cnt = 0;
  size_t vec_cnt = 0;
  for (const auto& [k, v] : map) {
    if (v->is<uint64_t>()) {
      uint_cnt++;
    }
    if (v->is<std::string>()) {
      str_cnt++;
    }
    if (v->is<std::vector<uint32_t>>()) {
      vec_cnt++;
    }
  }

  ASSERT_EQ(uint_cnt, 1);
  ASSERT_EQ(str_cnt, 1);
  ASSERT_EQ(vec_cnt, 1);
}

}  // namespace thirdai::ar::tests