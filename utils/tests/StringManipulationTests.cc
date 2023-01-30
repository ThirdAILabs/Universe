#include <gtest/gtest.h>
#include <utils/StringManipulation.h>

namespace thirdai::text {

void assertSplitSentence(std::vector<std::string_view>& words) {
  ASSERT_EQ(words.size(), 7);
  ASSERT_EQ(words[0], "This");
  ASSERT_EQ(words[1], "is");
  ASSERT_EQ(words[2], "a");
  ASSERT_EQ(words[3], "sentence");
  ASSERT_EQ(words[4], "with");
  ASSERT_EQ(words[5], "many");
  ASSERT_EQ(words[6], "words.");
}

TEST(StringManipulationTest, TestSplitWithDelimiter) {
  std::string_view sentence = "This is a sentence with many words.";
  auto words = split(sentence);
  assertSplitSentence(words);

  sentence = "This-is-a-sentence-with-many-words.";
  words = split(sentence, '-');
  assertSplitSentence(words);
}

TEST(StringManipulationTest, TestSplitStartAndEndWithDelimiter) {
  std::string_view sentence = " It's funky time. ";
  auto words = split(sentence);
  ASSERT_EQ(words[0], "It's");
  ASSERT_EQ(words[1], "funky");
  ASSERT_EQ(words[2], "time.");
}

TEST(StringManipulationTest, TestSplitConsecutiveDelimiter) {
  std::string_view sentence = "It's funky  time.";
  auto words = split(sentence);
  ASSERT_EQ(words[0], "It's");
  ASSERT_EQ(words[1], "funky");
  ASSERT_EQ(words[2], "time.");
}

}  // namespace thirdai::text