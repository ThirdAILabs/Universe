#include <gtest/gtest.h>
#include <archive/src/Archive.h>
#include <archive/src/StringCipher.h>
#include <archive/src/Value.h>
#include <archive/tests/Utils.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::ar::tests {

template <typename T, typename U>
void testValue(T value) {
  ConstArchivePtr archive = Value<T>::make(value);

  ASSERT_EQ(archive->as<T>(), value);

  ASSERT_TRUE(archive->is<T>());
  ASSERT_FALSE(archive->is<U>());

  CHECK_EXCEPTION(archive->as<U>(),
                  "Attempted to convert archive of type '" +
                      Value<T>::typeName() + "' to type '" +
                      Value<U>::typeName() + "'.",
                  std::runtime_error)

  std::stringstream buffer;
  serialize(archive, buffer);
  auto loaded = deserialize(buffer);

  ASSERT_EQ(loaded->as<T>(), value);

  ASSERT_TRUE(loaded->is<T>());
  ASSERT_FALSE(loaded->is<U>());

  CHECK_EXCEPTION(loaded->as<U>(),
                  "Attempted to convert archive of type '" +
                      Value<T>::typeName() + "' to type '" +
                      Value<U>::typeName() + "'.",
                  std::runtime_error)
}

TEST(ValueTests, TestBoolean) {
  testValue<bool, uint64_t>(true);
  testValue<bool, uint64_t>(false);
}

TEST(ValueTests, TestUint64) {
  testValue<uint64_t, float>(720942);
  testValue<uint64_t, float>(824024902);
}

TEST(ValueTests, TestInt64) {
  testValue<int64_t, uint64_t>(-720942);
  testValue<int64_t, uint64_t>(824024902);
}

TEST(ValueTests, TestFloat) {
  testValue<float, uint64_t>(-2.04924);
  testValue<float, uint64_t>(8.32902);
}

TEST(ValueTests, TestString) {
  testValue<std::string, uint64_t>("apple");
  testValue<std::string, uint64_t>("banana");
}

TEST(ValueTests, StringValuesAreHidden) {
  auto str_archive = str("pineapple");

  std::stringstream buffer;
  serialize(str_archive, buffer);

  std::string serialized = buffer.str();

  ASSERT_EQ(serialized.find("pineapple"), std::string::npos);

  ASSERT_NE(serialized.find(cipher("pineapple")), std::string::npos);
}

TEST(ValueTests, TestVecUint32) {
  testValue<std::vector<uint32_t>, uint64_t>({1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(ValueTests, TestVecInt64) {
  testValue<std::vector<int64_t>, int64_t>({1, -2, 3, -4, 5, -6, 7, -8});
}

TEST(ValueTests, TestVecString) {
  testValue<std::vector<std::string>, uint64_t>({"abc", "bcd", "cde"});
}

TEST(ValueTests, TestVecWstring) {
  testValue<std::vector<std::wstring>, uint64_t>({L"abc", L"bcd", L"cde"});
}

TEST(ValueTests, TestMapUint64) {
  testValue<std::unordered_map<uint64_t, std::vector<uint64_t>>, uint64_t>(
      {{1, {2, 3}}, {2, {3, 4}}, {3, {4, 5}}});
}

TEST(ValueTests, TestMapFloat) {
  testValue<std::unordered_map<uint64_t, std::vector<float>>, uint64_t>(
      {{1, {2.0, -3.0}}, {2, {-3.1, 4.45}}, {3, {0.042, 5.42}}});
}

}  // namespace thirdai::ar::tests