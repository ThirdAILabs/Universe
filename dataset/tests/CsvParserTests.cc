#include <gtest/gtest.h>
#include <dataset/src/utils/CsvParser.h>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace thirdai::dataset::tests {

void testCsvParser(const std::string& input_string, char delimiter,
                   const std::vector<std::string>& expected_parsed_result) {
  auto parsed_result = parsers::CSV::parseLine(input_string, delimiter);
  ASSERT_EQ(parsed_result.size(), expected_parsed_result.size());
  for (uint32_t i = 0; i < parsed_result.size(); i++) {
    ASSERT_EQ(std::string(parsed_result[i]), expected_parsed_result[i]);
  }
}

TEST(CsvParserTests, SupportsAnySingleCharDelimiter) {
  testCsvParser("A,B", ',', {"A", "B"});
  testCsvParser("A\tB", '\t', {"A", "B"});
  testCsvParser("A:B", ':', {"A", "B"});
}

TEST(CsvParserTests, HandlesMultipleColumns) {
  testCsvParser("A,B,C,D", ',', {"A", "B", "C", "D"});
}

TEST(CsvParserTests, HandlesEmptyColumns) {
  testCsvParser("", ',', {""});
  testCsvParser("A,,C,D", ',', {"A", "", "C", "D"});
  testCsvParser(",B,C,D", ',', {"", "B", "C", "D"});
  testCsvParser("A,B,C,", ',', {"A", "B", "C", ""});
}

TEST(CsvParserTests, HandlesMultipleEmptyColumns) {
  testCsvParser("A,,,D", ',', {"A", "", "", "D"});
  testCsvParser(",B,C,", ',', {"", "B", "C", ""});
  testCsvParser("A,,C,", ',', {"A", "", "C", ""});
  testCsvParser(",,,", ',', {"", "", "", ""});
}

TEST(CsvParserTests, RemovesDoubleQuotesAroundStringColumn) {
  testCsvParser("\"string column\"", ',', {"string column"});
}

TEST(CsvParserTests, IgnoresDelimiterInStringColumn) {
  testCsvParser("\"I wish there was no delimiter, but there is.\"", ',',
                {"I wish there was no delimiter, but there is."});
}

TEST(CsvParserTests, DoubleQuotesInStringColumnNotMistakenAsOuterQuotes) {
  // CSV parsers distinguish between double quotes that wrap around a string
  // and double quotes inside of the string by duplicating the inner quotes.
  // Typically, the CSV parser would deduplicate the double quotes by
  // writing a completely new string, but we don't do that and instead return a
  // a substring for performance reasons.
  // The performance hit probably isn't too bad, but we can't do that until we
  // change all instances of string_view to regular strings
  // TODO(Geordie): Check overhead of switching to regular strings and removing
  // extraneous characters during CSV parsing.
  testCsvParser(R"("""This is in quotes"", and this is not.")", ',',
                {R"(""This is in quotes"", and this is not.)"});
}

TEST(CsvParserTests, EscapeCharactersAreIgnored) {
  testCsvParser(R"(A\,B)", ',', {"A\\", "B"});
  // Since the inner escape character is ignored, the first double-quote
  // inside quotes is not ignored, thus the pair of double quotes is considered
  // to be quotes inside a quoted column, as opposed to closing quotes.
  testCsvParser(R"("A\"",B")", ',', {R"(A\"",B)"});
  // Inner quotes are not ignored, so the first inner quote is treated like an
  // end quote, meaning that the quotes are malformed. Thus, the outer quotes
  // are not trimmed.
  testCsvParser(R"("\"ABC\"")", ',', {R"("\"ABC\"")"});
  // Parser expects that the line does not contain unquoted newline character in
  // the middle of the line.
  // NOLINTNEXTLINE since clang-tidy doesn't like ASSERT_THROW
  ASSERT_THROW(testCsvParser("A\\\nB,C", ',', {}), std::invalid_argument);
}

TEST(CsvParserTests, BadDelimiterThrows) {
  // NOLINTNEXTLINE since clang-tidy doesn't like ASSERT_THROW
  ASSERT_THROW(testCsvParser("", '\\', {}), std::invalid_argument);
  // NOLINTNEXTLINE since clang-tidy doesn't like ASSERT_THROW
  ASSERT_THROW(testCsvParser("", '\"', {}), std::invalid_argument);
  // NOLINTNEXTLINE since clang-tidy doesn't like ASSERT_THROW
  ASSERT_THROW(testCsvParser("", '\n', {}), std::invalid_argument);
  // NOLINTNEXTLINE since clang-tidy doesn't like ASSERT_THROW
  ASSERT_THROW(testCsvParser("", '\r', {}), std::invalid_argument);
}

TEST(CsvParserTests, UnquotedNewlineInMiddleOfLineThrows) {
  // NOLINTNEXTLINE since clang-tidy doesn't like ASSERT_THROW
  ASSERT_THROW(testCsvParser("A\nB", ',', {}), std::invalid_argument);
}

TEST(CsvParserTests, UnquotedNewlineAtEndOfLineIsTrimmed) {
  testCsvParser("\n", ',', {""});
  testCsvParser("A\n", ',', {"A"});
  testCsvParser("A,B,C\n", ',', {"A", "B", "C"});
}

/*
  TESTS FOR MALFORMED QUOTES
  If quoted column is malformed and we have seen delimiters inside the
  quotes, treat the first delimiter in quotes as an unquoted delimiter.
  Quoted column is malformed if we reach end of line without seeing an end
  quote or if and end quote is followed by a regular character.
*/

TEST(CsvParserTests, RegularCharactersAfterEndQuote) {
  // When there are regular characters after the end quote, then we treat
  // the column as an unquoted column.
  testCsvParser("\"I wish there was no\" delimiter, but there is.", ',',
                {"\"I wish there was no\" delimiter", " but there is."});
  testCsvParser("\"There is no\" delimiter", ',',
                {"\"There is no\" delimiter"});
}

TEST(CsvParserTests, NoEndQuote) {
  // No end quote. Treat first delimiter after the opening quote as
  // end of first column.
  testCsvParser("\"I wish there was no delimiter, but there is.", ',',
                {"\"I wish there was no delimiter", " but there is."});
  // Properly handles multiple delimiters after opening quote.
  testCsvParser("\"I wish there was no delimiter, but there is,one", ',',
                {"\"I wish there was no delimiter", " but there is", "one"});
}

TEST(CsvParserTests, TrimsReturnAndNewlineCharacters) {
  testCsvParser("Hello\r", ',', {"Hello"});
  testCsvParser("Hello\r\n", ',', {"Hello"});
  testCsvParser("Hello\n", ',', {"Hello"});
  testCsvParser("\r", ',', {""});
  testCsvParser("\r\n", ',', {""});
  testCsvParser("\n", ',', {""});
}

TEST(CsvParserTests, QuotesInQuotes) {
  testCsvParser(R"("Hey there"", Delilah")", ',', {R"(Hey there"", Delilah)"});
  testCsvParser(R"("Hey, ""Delilah""")", ',', {R"(Hey, ""Delilah"")"});
}

}  // namespace thirdai::dataset::tests