#include <gtest/gtest.h>
#include <utils/src/BloomFilter.h>
#include <string>

TEST(BloomFilterTest, insert) {
  thirdai::utils::BloomFilter<std::string> bloom(100, 0.01);
  for (int i = 0; i < 100; i++) {
    bloom.add(std::to_string(i));
  }
  ASSERT_TRUE(bloom.size() <= 100);
  ASSERT_TRUE(bloom.size() > 50);
}

TEST(BloomFilterTest, insert_and_contains) {
  thirdai::utils::BloomFilter<std::string> bloom(100, 0.01);
  for (int i = 0; i < 100; i++) {
    bloom.add(std::to_string(i));
  }
  for (int i = 0; i < 100; i++) {
    ASSERT_TRUE(bloom.contains(std::to_string(i)));
  }
}