#include <gtest/gtest.h>
#include <serialization/src/Archive.h>
#include <serialization/src/ArchiveList.h>
#include <serialization/src/ArchiveMap.h>
#include <serialization/tests/Utils.h>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::ar::tests {

TEST(ArchiveTests, Serialization) {
  auto archive = ArchiveMap::make();

  auto list = ArchiveList::make();
  list->append(str("abc"));
  std::vector<uint32_t> vec_val = {1, 2, 3, 4};
  list->append(vec(vec_val));

  archive->at("list") = list;
  archive->at("int") = u64(8642);

  auto sub_archive = ArchiveMap::make();
  std::unordered_map<uint64_t, std::vector<float>> map_val = {
      {1, {2.0, 3.0}}, {10, {20.0, 30.0}}};
  sub_archive->at("map") = map(map_val);

  archive->at("sub") = sub_archive;

  std::stringstream buffer;
  serialize(archive, buffer);
  auto loaded = deserialize(buffer);

  CHECK_EXCEPTION(loaded->get("x"), "Map contains no value for key 'x'.",
                  std::out_of_range);
  ASSERT_FALSE(loaded->contains("x"));

  ASSERT_EQ(loaded->get("list")->list().at(0)->as<std::string>(), "abc");
  ASSERT_EQ(loaded->get("list")->list().at(1)->as<std::vector<uint32_t>>(),
            vec_val);

  CHECK_EXCEPTION(loaded->get("list")->list().at(0)->as<float>(),
                  "Attempted to convert archive of type 'Value[std::string]' "
                  "to type 'Value[float]'.",
                  std::runtime_error)

  ASSERT_EQ(loaded->getAs<uint64_t>("int"), 8642);
  ASSERT_EQ(loaded->getOpt<std::string>("x"), std::nullopt);
  ASSERT_EQ(loaded->getOr<std::string>("x", "xyz"), "xyz");

  auto loaded_map_val =
      loaded->get("sub")
          ->getAs<std::unordered_map<uint64_t, std::vector<float>>>("map");
  ASSERT_EQ(loaded_map_val, map_val);

  ASSERT_FALSE(loaded->get("sub")->contains("a"));
}

}  // namespace thirdai::ar::tests