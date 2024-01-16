#include "CppClassifier.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

std::vector<std::string> split(const std::string& line, char delimiter) {
  std::vector<std::string> items;
  size_t start = 0, end;
  while ((end = line.find_first_of(delimiter, start)) != std::string::npos) {
    items.push_back(line.substr(start, end - start));
    start = end + 1;
  }
  if (start < line.size()) {
    items.push_back(line.substr(start));
  }
  return items;
}

int main(int argc, const char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: ./fraud_detection_eval <saved_model> <test_file>"
              << std::endl;
    return 1;
  }

  thirdai::licensing::activate("<license key here>");

  auto model = thirdai::CppClassifier::load(argv[1]);

  uint32_t correct = 0, total = 0;

  std::ifstream test_file(argv[2]);

  std::string line;
  std::getline(test_file, line);
  auto columns = split(line, ',');
  size_t label_column = std::distance(
      columns.begin(), std::find(columns.begin(), columns.end(), "isFraud"));

  while (std::getline(test_file, line)) {
    auto values = split(line, ',');

    std::unordered_map<std::string, std::string> input;
    for (size_t i = 0; i < columns.size(); i++) {
      if (i != label_column) {
        input[columns.at(i)] = values.at(i);
      }
    }

    uint32_t label = std::stoul(values.at(label_column));

    uint32_t pred = model->predict(input);

    if (pred == label) {
      correct++;
    }
    total++;
  }

  std::cerr << "Accuracy=" << static_cast<float>(correct) / total << std::endl;

  return 0;
}