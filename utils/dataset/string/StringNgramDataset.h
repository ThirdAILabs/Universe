#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "Dataset.h"
#include "MurmurHash3.h"

namespace thirdai::utils {

struct Ngram {
  std::string _gram;
  // The stuff below are for if we want to use a recursive hash function.
  std::string _passed_prefix;
  std::string _new_suffix;
};


class CharStream {
  public:
  virtual const Ngram &nextNGram() = 0;

};

/**
 * For set of strings?
 * Set of documents?
 */
class StringNgramDataset: public Dataset {

  public:
  void vectorize(const std::string& str, uint32_t n) {
    void *start = (void*) str.c_str();
    size_t len = str.length();
    std::vector<uint32_t> hashes;
    for (size_t i = 0; i < len - n + 1; i++) {
      uint32_t hash;
      MurmurHash3_x86_32(start + i, n * sizeof(char), 341, (void *) &hash);
      hashes.push_back(hash);
    }
    std::sort(hashes.begin(), hashes.end());
    
  };
  

};

}  // namespace thirdai::utils