#pragma once

#include <hashing/src/MurmurHash.h>
#include <dataset/src/string/loaders/StringLoader.h>
#include <dataset/src/string/vectorizers/CompositeVectorizer.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::dataset {

class GlobalFreq {
  /*
      This class is responsible for calculating (inverse) word/document
     frequencies on a given data corpus.
  */
  // TODO(Henry): documentation

 public:
  GlobalFreq(std::unique_ptr<StringLoader> string_loader,
             const vectorizer_config_t& vectorizer_config,
             std::vector<std::string>& filenames);

  // Returns a copy of _idf_map.
  std::unordered_map<uint32_t, float> getIdfMap() const { return _idf_map; }

  vectorizer_config_t getVectorizerConfig() const { return _vectorizer_config; }

  ~GlobalFreq();

 private:
  std::unique_ptr<StringLoader> _string_loader;
  vectorizer_config_t _vectorizer_config;
  /**
   * Map from feature hash to inverse document frequency
   */
  std::unordered_map<uint32_t, float> _idf_map;
};

}  // namespace thirdai::dataset