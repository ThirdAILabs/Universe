#include <gtest/gtest.h>
#include <utils/StringManipulation.h>

namespace thirdai::utils {

TEST(StringManipulationTest, TestSplitIntoWords) {
  std::string_view sentence = "This is a sentence with many words.";
  auto words = token_encoding::splitIntoWords(sentence);
  ASSERT_EQ(words.size(), 7);
  ASSERT_EQ(words[3], "sentence");

  sentence = "This-is-a-sentence-with-many-words.";
  words = token_encoding::splitIntoWords(sentence, '-');
  ASSERT_EQ(words.size(), 7);
  ASSERT_EQ(words[3], "sentence");
}

}  // namespace thirdai::utils