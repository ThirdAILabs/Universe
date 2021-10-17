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
template <typename Loader_t>
class GlobalFreq {
  /*
      This class is responsible for calculating (inverse) word/document
     frequencies on a given data corpus.
  */
  // TODO(Henry): documentation
 private:
  Loader_t _string_loader;
  CompositeVectorizer _vectorizer;

 public:
  /**
   * Map from feature hash to inverse document frequency
   */
  std::unordered_map<uint32_t, float> _idfMap;
  GlobalFreq(vectorizer_config_t config, std::vector<std::string>&& filenames);

  /**
   * The return value can be compared with the hash of the vectorizer config
   * used by StringFactory to ensure that string factory uses the right global
   * frequencies object for its vectorizer.
   */
  uint32_t getVectorizerConfigHash() const {
    return _vectorizer.getConfigHash();
  }

  ~GlobalFreq();
};

}  // namespace thirdai::utils