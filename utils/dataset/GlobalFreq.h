#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::utils {

class GlobalFreq {
  /*
      This class is responsible for calculating (inverse) word/document
     frequencies on a given data corpus.
  */
 private:
  std::unordered_map<std::string, int> _idfMap;

 public:
  GlobalFreq(std::vector<std::string> files);

  inline int getIdf(std::string word);  // Should have a default value

  // Can have a parallel version of getIdf like:
  void getIdfPar(std::vector<std::string> wordvec, int* freqs);

  ~GlobalFreq();
};

GlobalFreq::GlobalFreq(std::vector<std::string> files) {}

GlobalFreq::~GlobalFreq() {}

}  // namespace thirdai::utils
