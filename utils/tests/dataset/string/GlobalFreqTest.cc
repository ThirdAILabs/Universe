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
  std::string query = "biden";
  std::cout << "IDF size: " << global_freq.idf_size() << std::endl;
  for (auto kv : global_freq._idfMap) {
    std::cout << kv.second << std::endl;
  }
}

}  // namespace thirdai::utils