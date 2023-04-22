#include <gtest/gtest.h>
#include <dataset/src/DataSource.h>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace thirdai::dataset::tests {

void testCsvDataSource(const std::string& input_string, char delimiter,
                       const std::vector<std::string>& expected_lines) {
  std::ofstream out("temp.csv");
  out << input_string;
  out.close();

  auto file_data_source = FileDataSource::make("temp.csv");

  auto csv_data_source = CsvDataSource::make(file_data_source, delimiter);
  for (const auto& line : expected_lines) {
    ASSERT_EQ(csv_data_source->nextLine().value(), line);
  }
  ASSERT_EQ(csv_data_source->nextLine(), std::nullopt);
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
}  // namespace thirdai::dataset::tests