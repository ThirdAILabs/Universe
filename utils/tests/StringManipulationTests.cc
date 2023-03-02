#include <gtest/gtest.h>
#include <utils/StringManipulation.h>
#include <string_view>

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
  auto words = split(sentence, /* delimiter= */ ' ');
  assertSplitSentence(words);

  sentence = "This-is-a-sentence-with-many-words.";
  words = split(sentence, '-');
  assertSplitSentence(words);
}

TEST(StringManipulationTest, TestSplitStartAndEndWithDelimiter) {
  std::string_view sentence = " It's funky time. ";
  auto words = split(sentence, /* delimiter= */ ' ');
  ASSERT_EQ(words[0], "It's");
  ASSERT_EQ(words[1], "funky");
  ASSERT_EQ(words[2], "time.");
}

TEST(StringManipulationTest, TestSplitConsecutiveDelimiter) {
  std::string_view sentence = "It's funky  time.";
  auto words = split(sentence, /* delimiter= */ ' ');
  ASSERT_EQ(words[0], "It's");
  ASSERT_EQ(words[1], "funky");
  ASSERT_EQ(words[2], "time.");
}

void assertEqualTokens(const std::vector<std::string_view>& parsed,
                       const std::vector<std::string>& expected) {
  ASSERT_EQ(parsed.size(), expected.size());
  for (uint32_t i = 0; i < parsed.size(); i++) {
    ASSERT_EQ(parsed[i], expected[i]);
  }
}

TEST(StringManipulationTest, TestTokenizeSentenceNoPunctuation) {
  std::string_view sentence = "no punctuation no cry";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"no", "punctuation", "no", "cry"});
}

TEST(StringManipulationTest, TestTokenizeSentencePunctuationAfterWord) {
  std::string_view sentence = "no punctuation? no way";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"no", "punctuation", "?", "no", "way"});
}

TEST(StringManipulationTest, TestTokenizeSentencePunctuationBetweenWords) {
  std::string_view sentence = "some:fake:coordinates";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"some", ":", "fake", ":", "coordinates"});
}

TEST(StringManipulationTest, TestTokenizeSentenceConsecutivePunctuations) {
  std::string_view sentence = "These are all punctuations: ?+(_)";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"These", "are", "all", "punctuations", ":", "?",
                            "+", "(", "_", ")"});
}

TEST(StringManipulationTest, TestTokenizeSentenceSurroundedByPunctuations) {
  std::string_view sentence = "\"Surrounded by quotes\"";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"\"", "Surrounded", "by", "quotes", "\""});
}

TEST(StringManipulationTest, TestTokenizeSentenceHorribleFormatting) {
  std::string_view sentence = "  ?  \"?!Surrounded  by::spaces  too?!\"  !  ";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"?", "\"", "?", "!", "Surrounded", "by", ":", ":",
                            "spaces", "too", "?", "!", "\"", "!"});
}

}  // namespace thirdai::text