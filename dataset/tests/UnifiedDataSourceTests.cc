#include <gtest/gtest.h>
#include <dataset/src/DataSource.h>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>

namespace thirdai::dataset::tests {
bool isInteger(const std::string& str) {
  return std::all_of(str.begin(), str.end(),
                     [](char c) { return std::isdigit(c) || c == '\n'; });
}

std::pair<std::string, std::string> writeNumbersAndAlphabetToFile() {
  std::string file_path_1 = "numbers_data.txt";
  std::ofstream file1(file_path_1);

  if (!file1.is_open()) {
    std::cerr << "Error opening file: " << file_path_1 << std::endl;
  }

  for (int i = 1; i <= 20; ++i) {
    file1 << i << std::endl;
  }

  file1.close();

  // Writing alphabet a to z in file2
  std::string file_path_2 = "alphabet_data.txt";
  std::ofstream file2(file_path_2);

  if (!file2.is_open()) {
    std::cerr << "Error opening file: " << file_path_2 << std::endl;
  }

  for (char c = 'a'; c <= 'z'; ++c) {
    file2 << c << std::endl;
  }

  file2.close();
  return {file_path_1, file_path_2};
}

TEST(UnifiedDataSourceTests, TestDataLoadSplit) {
  auto [file_path_1, file_path_2] = writeNumbersAndAlphabetToFile();

  auto unified_data_source = UnifiedDataSource::make(
      {FileDataSource::make(file_path_1), FileDataSource::make(file_path_2)},
      {0.2, 0.8}, 0);

  uint32_t number_data_points = 0, character_data_points = 0;
  while (auto line_data = unified_data_source->nextLine()) {
    // loop wont execute if line_data doesnt have value
    if (isInteger(*line_data)) {
      number_data_points++;
    } else {
      character_data_points++;
    }
  }
  ASSERT_LT(number_data_points,
            int(0.3 * (number_data_points + character_data_points)));
  ASSERT_GT(character_data_points,
            int(0.7 * (number_data_points + character_data_points)));
}
}  // namespace thirdai::dataset::tests