#include <gtest/gtest.h>
#include <utils/text/RegexPatterns.h>
#include <utility>

namespace thirdai::text::tests {

void runSingleRegexTest(const std::string& input,
                        const std::string& expected_output,
                        const RegexSub& regex_sub) {
  auto [regex, substitution] = regex_sub;
  std::string output = std::regex_replace(input, regex, substitution);
  ASSERT_EQ(output, expected_output);
}

void runRegexTest(const std::vector<std::pair<std::string, std::string>>&
                      input_expected_output_pairs,
                  const RegexSub& regex_sub) {
  for (auto [input, expected_output] : input_expected_output_pairs) {
    runSingleRegexTest(input, expected_output, regex_sub);
  }
}

TEST(RegexPatternsTest, SurroundNonDigitColonOrComma) {
  runRegexTest(
      {{"This: is", "This :  is"}, {"Note,important", "Note , important"}},
      SURROUND_NON_DIGIT_COLON_OR_COMMA);
}

TEST(RegexPatternsTest, SurroundEndLineColonOrComma) {
  runRegexTest({{"End of line:", "End of line : "}, {"Finish,", "Finish , "}},
               SURROUND_END_LINE_COLON_OR_COMMA);
}

TEST(RegexPatternsTest, SurroundElipses) {
  runRegexTest({{"Wait...", "Wait ... "}, {"So..", "So .. "}},
               SURROUND_ELIPSES);
}

TEST(RegexPatternsTest, SurroundSpecialChars) {
  runRegexTest({{"Special@character", "Special @ character"},
                {"Is this a question?", "Is this a question ? "}},
               SURROUND_SPECIAL_CHARS);
}

TEST(RegexPatternsTest, SurroundEndApostrophe) {
  runRegexTest({{"Sams' Club?", "Sams ' Club?"}}, SURROUND_END_APOSTROPHE);
}

TEST(RegexPatternsTest, SurroundEndPeriod) {
  runRegexTest({{"End.", "End . "}, {"Stop now.", "Stop now . "}},
               SURROUND_END_PERIOD);
}

TEST(RegexPatternsTest, SurroundParensBrackets) {
  runRegexTest(
      {{"(Parenthesis)", " ( Parenthesis ) "}, {"[Bracket]", " [ Bracket ] "}},
      SURROUND_PARENS_BRACKETS);
}

TEST(RegexPatternsTest, SurroundDoubleDashes) {
  runRegexTest({{"Word--word", "Word -- word"}}, SURROUND_DOUBLE_DASHES);
}

TEST(RegexPatternsTest, CompressMultipleSpace) {
  runRegexTest({{"Too  much   space", "Too much space"},
                {"Space    here", "Space here"}},
               COMPRESS_MULTIPLE_SPACE);
}

TEST(RegexPatternsTest, SurroundDoubleQuote) {
  runRegexTest({{"\"Quote\"", " \" Quote \" "},
                {"He said, \"Hello\"", "He said,  \" Hello \" "}},
               SURROUND_DOUBLE_QUOTE);
}

TEST(RegexPatternsTest, SurroundDoubleSingleQuote) {
  runRegexTest({{"This is ''important''", "This is  '' important '' "},
                {"''Quotes''", " '' Quotes '' "}},
               SURROUND_DOUBLE_SINGLE_QUOTE);
}

TEST(RegexPatternsTest, SurroundSpaceSingleQuote) {
  runRegexTest({{"A 'quote'", "A ' quote'"},
                {"'Start' of something", " ' Start' of something"}},
               SURROUND_SPACE_SINGLE_QUOTE);
}

TEST(RegexPatternsTest, SeparateContractions) {
  runRegexTest(
      {{"I'm happy", "I 'm  happy"}, {"You're right", "You 're  right"}},
      SEPARATE_CONTRACTIONS);
}

TEST(RegexPatternsTest, PeriodAfterSpace) {
  runRegexTest({{"Space . Not a number", "Space .  Not a number"},
                {"Another example .", "Another example . "}},
               PERIOD_AFTER_SPACE);
}

TEST(RegexPatternsTest, SurroundNonAbbreviationPeriods) {
  runRegexTest({{"U.S. 3.2 Mr. ", "U.S. 3.2 Mr . "}},
               SURROUND_NON_ABBREVIATION_PERIODS);
}

TEST(RegexPatternsTest, NltkWordTokenize) {
  std::string sentence =
      "don't think (the U.S.A., or Canada) for this--that they've 'seen a lot' "
      "of things it's 'mr. president.' WHat%l 12.6 66.889 3,000 the-thing \"hello";
  std::string expected_output =
      "do n't think ( the U.S.A. , or Canada ) for this -- that they 've ' "
      "seen a lot ' of things it 's ' mr . president . ' WHat % l 12.6 "
      "66.889 3,000 the-thing \" hello";

  ASSERT_EQ(nltkWordTokenize(sentence), expected_output);
}

}  // namespace thirdai::text::tests