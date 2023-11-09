#include <gtest/gtest.h>
#include <serialization/src/Archive.h>
#include <serialization/src/ArchiveValue.h>
#include <serialization/tests/Utils.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace thirdai::ar::tests {

template <typename T, typename U>
void testArchiveValue(T value) {
  ConstArchivePtr archive = ArchiveValue<T>::make(value);

  ASSERT_EQ(archive->as<T>(), value);

  ASSERT_TRUE(archive->is<T>());
  ASSERT_FALSE(archive->is<U>());

  CHECK_EXCEPTION(archive->as<U>(),
                  "Attempted to convert archive of type '" +
                      ArchiveValue<T>::typeName() + "' to type '" +
                      ArchiveValue<U>::typeName() + "'.",
                  std::runtime_error)

  std::stringstream buffer;
  serialize(archive, buffer);
  auto loaded = deserialize(buffer);

  ASSERT_EQ(loaded->as<T>(), value);

  ASSERT_TRUE(loaded->is<T>());
  ASSERT_FALSE(loaded->is<U>());

  CHECK_EXCEPTION(loaded->as<U>(),
                  "Attempted to convert archive of type '" +
                      ArchiveValue<T>::typeName() + "' to type '" +
                      ArchiveValue<U>::typeName() + "'.",
                  std::runtime_error)
}

TEST(ArchiveValueTests, TestBoolean) {
  testArchiveValue<bool, uint64_t>(true);
  testArchiveValue<bool, uint64_t>(false);
}

TEST(ArchiveValueTests, TestUint64) {
  testArchiveValue<uint64_t, float>(720942);
  testArchiveValue<uint64_t, float>(824024902);
}

TEST(ArchiveValueTests, TestInt64) {
  testArchiveValue<int64_t, uint64_t>(-720942);
  testArchiveValue<int64_t, uint64_t>(824024902);
}

TEST(ArchiveValueTests, TestFloat) {
  testArchiveValue<float, uint64_t>(-2.04924);
  testArchiveValue<float, uint64_t>(8.32902);
}

TEST(ArchiveValueTests, TestString) {
  testArchiveValue<std::string, uint64_t>("apple");
  testArchiveValue<std::string, uint64_t>("banana");
}

TEST(ArchiveValueTests, TestVecUint32) {
  testArchiveValue<std::vector<uint32_t>, uint64_t>({1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(ArchiveValueTests, TestVecInt64) {
  testArchiveValue<std::vector<int64_t>, int64_t>({1, -2, 3, -4, 5, -6, 7, -8});
}

TEST(ArchiveValueTests, TestVecString) {
  testArchiveValue<std::vector<std::string>, uint64_t>({"abc", "bcd", "cde"});
}

TEST(ArchiveValueTests, TestVecWstring) {
  testArchiveValue<std::vector<std::wstring>, uint64_t>(
      {L"abc", L"bcd", L"cde"});
}

TEST(ArchiveValueTests, TestMapUint64) {
  testArchiveValue<std::unordered_map<uint64_t, std::vector<uint64_t>>,
                   uint64_t>({{1, {2, 3}}, {2, {3, 4}}, {3, {4, 5}}});
}

TEST(ArchiveValueTests, TestMapFloat) {
  testArchiveValue<std::unordered_map<uint64_t, std::vector<float>>, uint64_t>(
      {{1, {2.0, -3.0}}, {2, {-3.1, 4.45}}, {3, {0.042, 5.42}}});
}

}  // namespace thirdai::ar::tests