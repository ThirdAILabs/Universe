#include "../../../dataset/string/GlobalFreq.h"
#include "../../../dataset/string/StringDataset.h"
#include <gtest/gtest.h>

namespace thirdai::utils {

std::string filename = "FreelandSep10_2020.txt";
std::vector<std::string> directory;
directory.push_back(filename);

TEST(GlobalFreqTest, ProcessFile) {
    SentenceLoader loader;
    GlobalFreq global_freq(directory, loader, 42);
    std::cout << global_freq.getIdf("Thank") << std::endl;
}

}  // namespace thirdai::utils