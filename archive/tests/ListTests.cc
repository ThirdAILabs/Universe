#include <gtest/gtest.h>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/tests/Utils.h>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace thirdai::ar::tests {

std::tuple<std::shared_ptr<List>, ConstArchivePtr, ConstArchivePtr,
           ConstArchivePtr>
simpleList() {
  auto list = List::make();

  auto a = u64(10);
  auto b = str("hello");
  auto c = vecU32({1, 2, 3});

  list->append(a);
  list->append(b);
  list->append(c);

  return {list, a, b, c};
}

TEST(ListTests, ListIndexing) {
  auto [list, a, b, c] = simpleList();

  ASSERT_EQ(list->size(), 3);

  ASSERT_EQ(list->at(0), a);
  ASSERT_EQ(list->at(1), b);
  ASSERT_EQ(list->at(2), c);

  CHECK_EXCEPTION(list->at(3), "Cannot access element 3 in list of size 3.",
                  std::out_of_range);
}

TEST(ListTests, ListIterator) {
  auto [list, a, b, c] = simpleList();

  std::vector<ConstArchivePtr> expected = {a, b, c};
  size_t cnt = 0;
  for (const auto& x : *list) {
    ASSERT_EQ(x, expected.at(cnt++));
  }
  ASSERT_EQ(cnt, 3);
}

TEST(ListTests, Serialization) {
  auto [list, a, b, c] = simpleList();

  std::stringstream buffer;
  serialize(list, buffer);
  auto loaded = deserialize(buffer);

  CHECK_EXCEPTION(loaded->map(),
                  "Expected to the archive to have type Map but found 'List'.",
                  std::runtime_error);

  ASSERT_EQ(loaded->list().size(), 3);

  ASSERT_EQ(loaded->list().at(0)->as<U64>(), a->as<U64>());
  ASSERT_EQ(loaded->list().at(1)->as<Str>(), b->as<Str>());
  ASSERT_EQ(loaded->list().at(2)->as<VecU32>(), c->as<VecU32>());

  std::unordered_set<ConstArchivePtr> visited;
  for (const auto& x : loaded->list()) {
    ASSERT_FALSE(visited.count(x));
    visited.insert(x);
  }

  ASSERT_EQ(visited.size(), 3);
}

}  // namespace thirdai::ar::tests