#include "CppClassifier.h"
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

int main(int argc, const char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: ./clinc_eval <saved_model> <test_file>" << std::endl;
    return 1;
  }

  thirdai::licensing::activate("<license key here>");

  auto model = thirdai::CppClassifier::load(argv[1]);

  uint32_t correct = 0, total = 0;

  std::ifstream test_file(argv[2]);
  std::string line;
  std::getline(test_file, line);  // Remove header
  while (std::getline(test_file, line)) {
    auto comma_loc = line.find_first_of(',');

    uint32_t label = std::stoul(line.substr(0, comma_loc));

    std::string text = (line.substr(comma_loc + 1));
    if (text.front() == '"' && text.back() == '"') {  // Strip quotes if present
      text = text.substr(1, text.size() - 2);
    }

    uint32_t pred = model->predict({{"text", text}});

    if (pred == label) {
      correct++;
    }
    total++;
  }

  std::cerr << "Accuracy=" << static_cast<float>(correct) / total << std::endl;

  return 0;
}