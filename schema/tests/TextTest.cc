#include <gtest/gtest.h>
#include <schema/Text.h>
#include <unordered_set>

namespace thirdai::schema {

TEST(CharacterNGramBlockTest, ProperlyHandlesOffset) {
  uint32_t offset = 0;
  uint32_t col = 0;
  CharacterNGramBlock::Builder(col, 3, 10000)->build(offset);
  CharacterNGramBlock::Builder(col, 3, 10000)->build(offset);
  CharacterNGramBlock::Builder(col, 3, 10000)->build(offset);
  ASSERT_EQ(offset, 30000);
}

TEST(WordNGramBlockTest, ProperlyHandlesOffset) {
  uint32_t offset = 0;
  uint32_t col = 0;
  WordNGramBlock::Builder(col, 3, 10000)->build(offset);
  WordNGramBlock::Builder(col, 3, 10000)->build(offset);
  WordNGramBlock::Builder(col, 3, 10000)->build(offset);
  ASSERT_EQ(offset, 30000);
}

TEST(CharacterNGramBlockTest, CorrectOutput) {
  uint32_t offset = 25;
  uint32_t col = 0;
  auto cb1 = CharacterNGramBlock::Builder(col, 3, 10000)->build(offset);
  auto cb2 = CharacterNGramBlock::Builder(col, 5, 10000)->build(offset);

  InProgressVector vec;
  std::string_view str = "some text here!";
  cb1->consume({str}, vec);
  cb2->consume({str}, vec);
  std::unordered_set<uint32_t> unique_seen;
  uint32_t i = 0;
  for (auto& kv : vec) {
    unique_seen.insert(kv.first);
    i++;
  }
  ASSERT_EQ(i, 24);

  // Should have barely any collisions.
  ASSERT_GE(unique_seen.size(), 24);

  bool seen_ge_10000 = false;
  bool seen_ge_20000 = false;
  for (const auto& seen : unique_seen) {
    if (seen >= 10000) {
      seen_ge_10000 = true;
    }
    if (seen >= 20000) {
      seen_ge_20000 = true;
    }
  }
  ASSERT_TRUE(seen_ge_10000);
  ASSERT_FALSE(seen_ge_20000);
}

TEST(WordNGramBlockTest, CorrectOutput) {
  uint32_t offset = 25;
  uint32_t col = 0;
  auto cb1 = WordNGramBlock::Builder(col, 2, 10000)->build(offset);
  auto cb2 = WordNGramBlock::Builder(col, 3, 10000)->build(offset);

  InProgressVector vec;
  std::string_view str = "Hello there!";
  cb1->consume({str}, vec);
  cb2->consume({str}, vec);
  std::unordered_set<uint32_t> unique_seen;
  uint32_t i = 0;
  for (auto& kv : vec) {
    unique_seen.insert(kv.first);
    i++;
  }
  ASSERT_EQ(i, 1);

  // Gut feeling for now. Should not be too many collisions.
  ASSERT_GE(unique_seen.size(), 1);

  bool seen_larger_than_10000 = false;
  for (const auto& seen : unique_seen) {
    if (seen > 10000) {
      seen_larger_than_10000 = true;
    }
  }
  ASSERT_FALSE(seen_larger_than_10000);
}


} // namespace thirdai::schema