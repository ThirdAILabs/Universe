#pragma once

#include "../../../hashing/MurmurHash.h"
#include "../loaders/StringLoader.h"
#include "../vectorizers/CompositeVectorizer.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::utils {
template <typename LOADER_T>
class GlobalFreq {
  /*
      This class is responsible for calculating (inverse) word/document
     frequencies on a given data corpus.
  */
  // TODO(Henry): documentation

 public:
  GlobalFreq(vectorizer_config_t config, std::vector<std::string>&& filenames);

  std::unordered_map<uint32_t, float> getIdfMap() const { return _idf_map; }

  vectorizer_config_t getVectorizerConfig() const {
    return _vectorizer.getConfig();
  }

  ~GlobalFreq();

 private:
  LOADER_T _string_loader;
  CompositeVectorizer _vectorizer;
  /**
   * Map from feature hash to inverse document frequency
   */
  std::unordered_map<uint32_t, float> _idf_map;
};

}  // namespace thirdai::utils