#include <gtest/gtest.h>
#include <dataset/src/utils/CsvParser.h>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace thirdai::dataset::tests {

// We accept arguments by value so we can pass literals.
// Makes it easier to write test cases, plus this isn't performance-critical.
// NOLINTNEXTLINE
void testCsvParser(std::string input_string, std::string delimiter,
                   std::vector<std::string> expected_parsed_result) {
  // TODO(Geordie): Implement
  CSV::verifyDelimiterIsValid(delimiter);
  auto parsed_result = CSV::parse(input_string, delimiter);
  ASSERT_EQ(parsed_result.size(), expected_parsed_result.size());
  for (uint32_t i = 0; i < parsed_result.size(); i++) {
    ASSERT_EQ(std::string(parsed_result[i]), expected_parsed_result[i]);
  }
}

TEST(CsvParserTests, SupportsArbitraryDelimiters) {
  testCsvParser("A,B", ",", {"A", "B"});
  testCsvParser("A\tB", "\t", {"A", "B"});
  testCsvParser("A::B", "::", {"A", "B"});
}

TEST(CsvParserTests, HandlesMultipleColumns) {
  testCsvParser("A,B,C,D", ",", {"A", "B", "C", "D"});
}

TEST(CsvParserTests, HandlesEmptyColumns) {
  testCsvParser("A,,C,D", ",", {"A", "", "C", "D"});
  testCsvParser(",B,C,D", ",", {"", "B", "C", "D"});
  testCsvParser("A,B,C,", ",", {"A", "B", "C", ""});
}

TEST(CsvParserTests, RemovesDoubleQuotesAroundStringColumn) {
  testCsvParser("\"string column\"", ",", {"string column"});
}

TEST(CsvParserTests, IgnoresDelimiterInStringColumn) {
  testCsvParser("\"I wish there was no delimiter, but there is.\"", ",",
                {"I wish there was no delimiter, but there is."});
}

TEST(CsvParserTests, DoubleQuotesInStringColumnNotMistakenAsOuterQuotes) {
  // CSV parsers distinguish between double quotes that wrap around a string
  // and double quotes inside of the string by duplicating the inner quotes.
  // Typically, the CSV parser would deduplicate the double quotes by
  // writing a completely new string, but we don't do that and instead return a
  // a substring for performance reasons.
  testCsvParser(R"("""This is in quotes"", and this is not.")", ",",
                {R"(""This is in quotes"", and this is not.)"});
}

TEST(CsvParserTests, BadDelimiterThrows) {
  // NOLINTNEXTLINE since clang-tidy doesn't like ASSERT_THROW
  ASSERT_THROW(testCsvParser("", "\\", {}), std::invalid_argument);
  // NOLINTNEXTLINE since clang-tidy doesn't like ASSERT_THROW
  ASSERT_THROW(testCsvParser("", "\"", {}), std::invalid_argument);
  // NOLINTNEXTLINE since clang-tidy doesn't like ASSERT_THROW
  ASSERT_THROW(testCsvParser("", "\n", {}), std::invalid_argument);
  // NOLINTNEXTLINE since clang-tidy doesn't like ASSERT_THROW
  ASSERT_THROW(testCsvParser("", "\r", {}), std::invalid_argument);
}

}  // namespace thirdai::dataset::tests