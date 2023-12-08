#include <gtest/gtest.h>
#include <utils/text/NeighboringCharacters.h>
#include <utils/text/StringManipulation.h>

namespace thirdai::text {

void assertSplitSentence(std::vector<std::string>& words) {
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
  std::string sentence = "This is a sentence with many words.";
  auto words = split(sentence, /* delimiter= */ ' ');
  assertSplitSentence(words);

  sentence = "This-is-a-sentence-with-many-words.";
  words = split(sentence, '-');
  assertSplitSentence(words);
}

TEST(StringManipulationTest, TestSplitStartAndEndWithDelimiter) {
  std::string sentence = " It's funky time. ";
  auto words = split(sentence, /* delimiter= */ ' ');
  ASSERT_EQ(words[0], "It's");
  ASSERT_EQ(words[1], "funky");
  ASSERT_EQ(words[2], "time.");
}

TEST(StringManipulationTest, TestSplitConsecutiveDelimiter) {
  std::string sentence = "It's funky  time.";
  auto words = split(sentence, /* delimiter= */ ' ');
  ASSERT_EQ(words[0], "It's");
  ASSERT_EQ(words[1], "funky");
  ASSERT_EQ(words[2], "time.");
}

void assertEqualTokens(const std::vector<std::string>& parsed,
                       const std::vector<std::string>& expected) {
  ASSERT_EQ(parsed.size(), expected.size());
  for (uint32_t i = 0; i < parsed.size(); i++) {
    ASSERT_EQ(parsed[i], expected[i]);
  }
}

TEST(StringManipulationTest, TestTokenizeSentenceTest) {
  std::string sentence = "abcde";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"abcde"});
}

TEST(StringManipulationTest, TestTokenizeSentenceNoPunctuation) {
  std::string sentence = "no punctuation no cry";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"no", "punctuation", "no", "cry"});
}

TEST(StringManipulationTest, TestTokenizeSentencePunctuationAfterWord) {
  std::string sentence = "no punctuation? no way";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"no", "punctuation", "?", "no", "way"});
}

TEST(StringManipulationTest, TestTokenizeSentencePunctuationBetweenWords) {
  std::string sentence = "some:fake:coordinates";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"some", ":", "fake", ":", "coordinates"});
}

TEST(StringManipulationTest, TestTokenizeSentenceConsecutivePunctuations) {
  std::string sentence = "These are all punctuations: ?+(_)";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"These", "are", "all", "punctuations", ":", "?",
                            "+", "(", "_", ")"});
}

TEST(StringManipulationTest, TestTokenizeSentenceSurroundedByPunctuations) {
  std::string sentence = "\"Surrounded by quotes\"";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"\"", "Surrounded", "by", "quotes", "\""});
}

TEST(StringManipulationTest, TestTokenizeSentenceHorribleFormatting) {
  std::string sentence = "  ?  \"?!Surrounded  by::spaces  too?!\"  !  ";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"?", "\"", "?", "!", "Surrounded", "by", ":", ":",
                            "spaces", "too", "?", "!", "\"", "!"});
}

TEST(StringManipulationTest, TestTokenizeSentenceNewLine) {
  std::string sentence = "Newline in the \n middle.";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"Newline", "in", "the", "middle", "."});

  sentence = "Return in the \r middle.";
  words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"Return", "in", "the", "middle", "."});

  sentence = "Return and newline in the \r\n middle.";
  words = tokenizeSentence(sentence);
  assertEqualTokens(words,
                    {"Return", "and", "newline", "in", "the", "middle", "."});
}

TEST(StringManipulationTest, TestTokenizeSentenceAccentedCharacters) {
  std::string sentence = "Soufflé";
  auto words = tokenizeSentence(sentence);
  assertEqualTokens(words, {"Soufflé"});
}

TEST(StringManipulationTest, CharKGramTest) {
  std::string sentence = "Some words";
  auto words = charKGrams(sentence, 4);
  assertEqualTokens(words,
                    {"Some", "ome ", "me w", "e wo", " wor", "word", "ords"});
}

TEST(StringManipulationTest, WordLevelCharKGramTest) {
  std::string sentence = "Some words";
  auto words = split(sentence, /* delimiter=*/' ');
  auto char_words = wordLevelCharKGrams(words, 4);
  assertEqualTokens(char_words, {"Some", "word", "ords"});

  char_words = wordLevelCharKGrams(words, 4, /* min_word_length=*/5);
  assertEqualTokens(char_words, {"word", "ords"});
}

TEST(StringManipulationTest, PerturbationReplaceWithSpace) {
  std::string test_str = "Hello";
  std::string result = replaceRandomCharactersWithSpaces(test_str, 2);
  int space_count = std::count(result.begin(), result.end(), ' ');
  ASSERT_EQ(space_count, 2);
}

TEST(StringManipulationTest, PerturbationDeleteCharacters) {
  std::string test_str = "Hello";
  std::string result = deleteRandomCharacters(test_str, 2);
  ASSERT_EQ(result.size(), 3);
}

TEST(StringManipulationTest, PerturbationReplaceWithAdjacentCharacters) {
  std::string test_str = "abcdef";
  std::string result = replaceRandomCharactersWithAdjacents(test_str, 6);
  std::cout << result << std::endl;

  bool all_replaced = true;
  for (int i = 0; i < result.size(); ++i) {
    if (std::find(neighboring_chars.at(test_str[i]).begin(),
                  neighboring_chars.at(test_str[i]).end(),
                  result[i]) == neighboring_chars.at(test_str[i]).end()) {
      all_replaced = false;
      break;
    }
  }
  ASSERT_TRUE(all_replaced);
}

TEST(StringManipulationTest, PerturbationDuplicateCharacters) {
  std::string test_str = "Hello";
  std::string result = duplicateRandomCharacters(test_str, 2);
  ASSERT_EQ(result.size(), test_str.size() + 2);
}

}  // namespace thirdai::text