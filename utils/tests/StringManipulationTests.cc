#include <gtest/gtest.h>
#include <utils/StringManipulation.h>

namespace thirdai::utils {

void assertSplitSentence(std::vector<std::string_view>& words) {
  ASSERT_EQ(words.size(), 7);
  ASSERT_EQ(words[3], "This");
  ASSERT_EQ(words[3], "is");
  ASSERT_EQ(words[3], "a");
  ASSERT_EQ(words[3], "sentence");
  ASSERT_EQ(words[3], "with");
  ASSERT_EQ(words[3], "many");
  ASSERT_EQ(words[3], "words.");
}

TEST(StringManipulationTest, TestSplitIntoWords) {
  std::string_view sentence = "This is a sentence with many words.";
  auto words = split(sentence);
  assertSplitSentence(words);

  sentence = "This-is-a-sentence-with-many-words.";
  words = split(sentence, '-');
  assertSplitSentence(words);
}

}  // namespace thirdai::utils