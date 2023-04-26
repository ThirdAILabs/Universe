#include <gtest/gtest.h>
#include <dataset/src/DataSource.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::dataset::tests {

void testCsvDataSource(const std::string& input_string, char delimiter,
                       const std::vector<std::string>& expected_lines) {
  const char* filename = "temp.csv";
  std::ofstream out(filename);
  out << input_string;
  out.close();

  auto file_data_source = FileDataSource::make(filename);

  auto csv_data_source = CsvDataSource::make(file_data_source, delimiter);
  for (const auto& line : expected_lines) {
    ASSERT_EQ(csv_data_source->nextLine().value(), line);
  }
  ASSERT_EQ(csv_data_source->nextLine(), std::nullopt);

  int result = std::remove(filename);
  if (result != 0) {
    std::cerr << "Failed to remove " << filename << std::endl;
  }
}

TEST(CsvDataSourceTests, HandlesQuotedNewline) {
  std::string input_string = "\"the first column\nhas a newline\"";
  testCsvDataSource(input_string, /* delimiter= */ ',',
                    {"\"the first column\nhas a newline\""});
}

TEST(CsvDataSourceTests, HandlesQuotedCarriageReturnAndNewline) {
  std::string input_string = "\"the first column\r\nhas a newline\"";
  testCsvDataSource(input_string, /* delimiter= */ ',',
                    {"\"the first column\r\nhas a newline\""});
}

TEST(CsvDataSourceTests, UnquotedNewlineOrCarriageReturn) {
  std::string input_string = "first line\nsecond line";
  testCsvDataSource(input_string, /* delimiter= */ ',',
                    {"first line", "second line"});
}

TEST(CsvDataSourceTests, MultipleColumns) {
  std::string input_string =
      "\"line 1\ncolumn 1\",\"line 1\ncolumn 2\"\n"
      "\"line 2\ncolumn 1\",\"line 2\ncolumn 2\"";
  testCsvDataSource(input_string, /* delimiter= */ ',',
                    {"\"line 1\ncolumn 1\",\"line 1\ncolumn 2\"",
                     "\"line 2\ncolumn 1\",\"line 2\ncolumn 2\""});
}

TEST(CsvDataSourceTests, NonCommaDelimiter) {
  std::string input_string =
      "\"line 1\ncolumn 1\"\t\"line 1\ncolumn 2\"\n"
      "\"line 2\ncolumn 1\"\t\"line 2\ncolumn 2\"";
  testCsvDataSource(input_string, /* delimiter= */ '\t',
                    {"\"line 1\ncolumn 1\"\t\"line 1\ncolumn 2\"",
                     "\"line 2\ncolumn 1\"\t\"line 2\ncolumn 2\""});
}

TEST(CsvDataSourceTests, EndsWithNewline) {
  std::string input_string =
      "\"line 1\ncolumn 1\"\t\"line 1\ncolumn 2\"\n"
      "\"line 2\ncolumn 1\"\t\"line 2\ncolumn 2\"\n";
  testCsvDataSource(input_string, /* delimiter= */ '\t',
                    {"\"line 1\ncolumn 1\"\t\"line 1\ncolumn 2\"",
                     "\"line 2\ncolumn 1\"\t\"line 2\ncolumn 2\""});
}

TEST(CsvDataSourceTests, MalformedQuotes) {
  std::string input_string = "\"the first column\nhas a newline";
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      testCsvDataSource(input_string, /* delimiter= */ ',', {}),
      std::invalid_argument);

  input_string = "\"the first\" column\nhas a newline";
  testCsvDataSource(input_string, /* delimiter= */ ',',
                    {"\"the first\" column", "has a newline"});
}

TEST(CsvDataSourceTests, NewlineNextToQuotes) {
  std::string input_string = "\"the first column\n\"";
  testCsvDataSource(input_string, /* delimiter= */ ',',
                    {"\"the first column\n\""});

  input_string = "\"\nhas a newline\"";
  testCsvDataSource(input_string, /* delimiter= */ ',',
                    {"\"\nhas a newline\""});

  input_string = "\"\n\"";
  testCsvDataSource(input_string, /* delimiter= */ ',', {"\"\n\""});
}

}  // namespace thirdai::dataset::tests