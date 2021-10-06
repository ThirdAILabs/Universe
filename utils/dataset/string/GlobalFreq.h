#pragma once

#include "../../hashing/MurmurHash.h"
//#include "StringDataset.h"
#include "vectorizers/StringVectorizer.h"
#include "loaders/StringLoader.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <math.h>

namespace thirdai::utils {

class GlobalFreq {
  /*
      This class is responsible for calculating (inverse) word/document
     frequencies on a given data corpus.
  */
 private:
  std::unordered_map<int, float> _idfMap;
  StringLoader* _string_loader;
  u_int32_t _murmur_seed;
  u_int32_t _max_dim = 100000;

 public:
  GlobalFreq(std::vector<std::string>& files, StringLoader* string_loader, u_int32_t murmur_seed);

  int getIdf(std::string& token);  // Should have a default value
  int getTF(std::string& token, std::string& doc);
  int getTokenID(std::string& token);
  int idf_size();

  // Can have a parallel version of getIdf like:
  void getIdfPar(std::vector<std::string> tokenvec, int* freqs);

  // ~GlobalFreq();
};

}  // namespace thirdai::utils