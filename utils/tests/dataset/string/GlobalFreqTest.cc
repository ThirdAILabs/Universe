#include "../../../dataset/string/GlobalFreq.h"
//#include "../../../dataset/string/StringDataset.h"
#include "../../../dataset/string/loaders/SentenceLoader.h"
#include "StringDatasetTest.cc"
#include <gtest/gtest.h>

namespace thirdai::utils {

// std::string filename = "FreelandSep10_2020.txt";

TEST(GlobalFreqTest, ProcessFile) {
  std::vector<std::string> directory;
  directory.push_back(filename);
  print_to_file();
  SentenceLoader loader;
  GlobalFreq global_freq(directory, &loader, 799);
  std::string query = "we";
  std::cout << global_freq.idf_size() << std::endl;
}

}  // namespace thirdai::utils