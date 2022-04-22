#include <gtest/gtest.h>
#include <schema/FeatureHashing.h>
#include <algorithm>
#include <vector>

namespace thirdai::schema {

TEST(FeatureHashingBlockTest, ProperlyHandlesOffset) {
  uint32_t offset = 0;
  FeatureHashingBlock::Builder(0, 1, 1)->build(offset);
  FeatureHashingBlock::Builder(0, 1, 1)->build(offset);
  FeatureHashingBlock::Builder(0, 1, 1)->build(offset);
  ASSERT_EQ(offset, 3);
}

TEST(FeatureHashingBlockTest, CorrectOutput) {
  uint32_t offset = 25;
  uint32_t col = 0;
  auto fhb1 = FeatureHashingBlock::Builder(col, 1, 10)->build(offset);
  auto fhb2 = FeatureHashingBlock::Builder(col, 3, 10)->build(offset);
InProgressVector vec;
  std::string_view num_str = "35";
  fhb1->consume({num_str}, vec);
  fhb2->consume({num_str}, vec);
  std::vector<uint32_t> seen;
  uint32_t i = 0;
  for (auto& kv : vec) {
    ASSERT_EQ(std::find(seen.begin(), seen.end(), kv.first), seen.end());
    ASSERT_EQ(kv.second, 1.0);
    if (i == 0) {
      ASSERT_LE(25, kv.first);
    } else {
      ASSERT_LE(35, kv.first);
      ASSERT_GT(45, kv.first);
    }
    seen.push_back(kv.first);
    i++;
  }
  ASSERT_EQ(i, 4);
}

} // namespace thirdai::schema