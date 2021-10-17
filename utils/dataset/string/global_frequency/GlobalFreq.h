#pragma once

#include "../../../hashing/MurmurHash.h"
#include "../loaders/StringLoader.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <cmath>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::utils {

class GlobalFreq {
  /*
      This class is responsible for calculating (inverse) word/document
     frequencies on a given data corpus.
  */
 // TODO(Henry): documentation
 private:
  StringLoader* _string_loader;
  u_int32_t _murmur_seed;
  u_int32_t _max_dim = 100000;

 public:
  std::unordered_map<uint32_t, float> _idfMap;
  GlobalFreq(std::vector<std::string>& files, StringLoader* string_loader,
             u_int32_t murmur_seed);

  int getIdf(std::string& token);  // Should have a default value
  
  int idf_size();
  // TODO(Henry): destructor
  // ~GlobalFreq();
};

}  // namespace thirdai::utils