#include <gtest/gtest.h>
#include <dataset/src/DataSource.h>
#include <fstream>
#include <string>
#include <vector>
namespace thirdai::dataset::tests {

void assertLinesAreEqual(const std::vector<std::string>& lines,
                         const std::vector<std::string>& expected_lines) {
  ASSERT_EQ(lines.size(), expected_lines.size());

  for (uint32_t i = 0; i < lines.size(); i++) {
    ASSERT_EQ(lines[i], expected_lines[i]);
  }
}

void testCsvParser(const std::string& csv_dataset, char delimiter,
                   const std::vector<std::string>& expected_lines) {
  static constexpr const char* filename = "csv_data_source_test_file.csv";
  std::ofstream out(filename);
  out << csv_dataset;
  out.close();

  auto file_source = FileDataSource::make(filename);
  CsvDataSource csv_source(file_source, delimiter);

  std::vector<std::string> lines;
  while (auto line = csv_source.nextLine()) {
    lines.push_back(*line);
  }

  assertLinesAreEqual(lines, expected_lines);

  csv_source.restart();
  lines.clear();

  auto batch_size = 3;

  while (auto batch = csv_source.nextBatch(batch_size)) {
    ASSERT_LE(batch->size(), batch_size);
    lines.insert(lines.end(), std::make_move_iterator(batch->begin()),
                 std::make_move_iterator(batch->end()));
  }

  assertLinesAreEqual(lines, expected_lines);
}

TEST(CsvDataSourceTests, SimpleLines) {
  const auto* dataset = "1\n2\n3\n4\n5\n6\n7";
  std::vector<std::string> expected_lines = {"1", "2", "3", "4", "5", "6", "7"};
  testCsvParser(dataset, ',', expected_lines);
}

TEST(CsvDataSourceTests, HandlesEscapes) {
  const auto* dataset =
      "1\\\na\n"
      "2\\\na\n"
      "3\\\na\n"
      "4\\\na\n"
      "5\\\na\n"
      "6\\\na\n"
      "7\\\na";
  std::vector<std::string> expected_lines = {
      "1\\\na", "2\\\na", "3\\\na", "4\\\na", "5\\\na", "6\\\na", "7\\\na"};
  testCsvParser(dataset, ',', expected_lines);
}

TEST(CsvDataSourceTests, HandlesQuotes) {
  const auto* dataset =
      "\"1\na\"\n"
      "\"2\na\"\n"
      "\"3\na\"\n"
      "\"4\na\"\n"
      "\"5\na\"\n"
      "\"6\na\"\n"
      "\"7\na\"";
  std::vector<std::string> expected_lines = {"\"1\na\"", "\"2\na\"", "\"3\na\"",
                                             "\"4\na\"", "\"5\na\"", "\"6\na\"",
                                             "\"7\na\""};
  testCsvParser(dataset, ',', expected_lines);
}

}  // namespace thirdai::dataset::tests