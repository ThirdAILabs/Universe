#include <gtest/gtest.h>
#include <schema/Number.h>

namespace thirdai::schema {

TEST(NumberBlockTest, ProperlyHandlesOffset) {
  uint32_t offset = 0;
  uint32_t col = 0;
  NumberBlock::Builder(col)->build(offset);
  NumberBlock::Builder(col)->build(offset);
  NumberBlock::Builder(col)->build(offset);
  ASSERT_EQ(offset, 3);
}

TEST(NumberBlockTest, CorrectOutput) {
  uint32_t offset = 25;
  uint32_t col = 0;
  auto nb = NumberBlock::Builder(col)->build(offset);
  InProgressVector vec;
  std::string_view num_str = "35";
  nb->consume({num_str}, vec);
  for (auto& kv : vec) {
    ASSERT_EQ(kv.first, 25);
    ASSERT_EQ(kv.second, 35.0);
  }
}

} // namespace thirdai::schema
