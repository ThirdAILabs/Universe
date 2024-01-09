#include <gtest/gtest.h>
#include <data/tests/MockDataSource.h>
#include <dataset/src/DataSource.h>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>

namespace thirdai::dataset::tests {
bool isInteger(const std::string& str) {
  return std::all_of(str.begin(), str.end(),
                     [](char c) { return std::isdigit(c); });
}

std::pair<std::vector<std::string>, std::vector<std::string>>
writeNumbersAndAlphabetToFile() {
  std::vector<std::string> numbers_data;

  for (int i = 1; i <= 20; ++i) {
    numbers_data.push_back(std::to_string(i));
  }

  std::vector<std::string> characters_data;
  for (char c = 'a'; c <= 'z'; ++c) {
    characters_data.push_back(std::string(1, c));
  }

  return {numbers_data, characters_data};
}

TEST(UnifiedDataSourceTests, TestDataLoadSplit) {
  auto [numbers_data, characters_data] = writeNumbersAndAlphabetToFile();

  std::set<std::string> numbers_data_set(numbers_data.begin(),
                                         numbers_data.end());
  std::set<std::string> characters_data_set(characters_data.begin(),
                                            characters_data.end());

  auto unified_data_source = UnifiedDataSource::make(
      {std::make_shared<data::tests::MockDataSource>(numbers_data),
       std::make_shared<data::tests::MockDataSource>(characters_data)},
      {0.2, 0.8}, 0);

  std::set<std::string> numbers_data_loaded;
  std::set<std::string> characters_data_loaded;
  uint32_t number_data_points = 0, character_data_points = 0;
  while (auto line_data = unified_data_source->nextLine()) {
    // loop wont execute if line_data doesnt have value
    if (isInteger(*line_data)) {
      number_data_points++;
      numbers_data_loaded.insert(*line_data);
    } else {
      character_data_points++;
      characters_data_loaded.insert(*line_data);
    }
  }

  ASSERT_GT(number_data_points, 10);
  ASSERT_GT(character_data_points, 10);
  ASSERT_EQ(characters_data_set, characters_data_loaded);
  ASSERT_EQ(numbers_data_set, numbers_data_loaded);
  ASSERT_LT(number_data_points,
            int(0.3 * (number_data_points + character_data_points)));
  ASSERT_GT(character_data_points,
            int(0.7 * (number_data_points + character_data_points)));
}
}  // namespace thirdai::dataset::tests