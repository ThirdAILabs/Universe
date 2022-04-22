#include <gtest/gtest.h>
#include <schema/DynamicCounts.h>
#include <schema/InProgressVector.h>
#include <schema/Schema.h>

namespace thirdai::schema {

TEST(DynamicCountsBlockTest, ProperlyHandlesOffset) {
  uint32_t id_col = 0;
  uint32_t timestamp_col = 1;
  int32_t target_col = 2;
  std::vector<Window> window_configs;
  window_configs.push_back({3, 3});
  std::string timestamp_fmt = "%Y-%b-%d";
  uint32_t offset = 0;
  DynamicCountsBlock::Builder(id_col, timestamp_col, target_col, window_configs, window_configs, timestamp_fmt)->build(offset);
  DynamicCountsBlock::Builder(id_col, timestamp_col, target_col, window_configs, window_configs, timestamp_fmt)->build(offset);
  DynamicCountsBlock::Builder(id_col, timestamp_col, target_col, window_configs, window_configs, timestamp_fmt)->build(offset);

  ASSERT_EQ(offset, 3);
}

TEST(DynamicCountsBlockTest, CorrectOutput) {
  
  uint32_t id_col = 0;
  uint32_t timestamp_col = 1;
  int32_t target_col = 2;
  std::vector<Window> window_configs;
  window_configs.push_back({3, 3});
  std::string timestamp_fmt = "%Y-%b-%d";
  uint32_t offset = 0;
  auto dcb = DynamicCountsBlock::Builder(id_col, timestamp_col, target_col, window_configs, window_configs, timestamp_fmt)->build(offset);
  
  InProgressVector output_vec;

  for (uint32_t i = 0; i < 7; i++) {
    std::stringstream date;
    date << "2014-Feb-0" << 1 + i; // For some reason it doesnt work with a trailing 0. grr...
    dcb->consume({"0", date.str(), "1"}, output_vec);
  }

  int32_t i = 0;
  for (const auto& iv : output_vec) {
    ASSERT_EQ(iv.first, 0);
    ASSERT_EQ(iv.second, std::max(0, i - 3));
    i++;
  }
  i = 0;
  for (const auto& label : output_vec.labels()) {
    ASSERT_EQ(label, std::max(0, i - 3));
    i++;
  }
  
}

} // namespace thirdai::schema