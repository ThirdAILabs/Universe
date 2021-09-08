#include "Dataset.h"
#include "GlobalFreq.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>

namespace thirdai::utils {

enum class TFIDF_FILE_TYPE{ FILE, SINGLE_SENTENCE, PARAGRAPH, N_SENTENCE };

class TfidfDataset : public Dataset {
 private:
  // TODO: Store a hash function(eg. murmurHash) that maps tokens to token ids
  // (indices), we can also use default natural numbers.

  int _tfDefault = -1;  // Specifies whether we should use a default value for
                        // the TF of a word.
  GlobalFreq* _globalFreq;

  // TODO: Also need a tokenization tool/object for word-unigram and
  // char-trigram;

 public:
  TfidfDataset(uint64_t target_batch_size, uint64_t target_batch_num_per_read,
               GlobalFreq* globalFreq, int tfDefault = -1)
      : Dataset(target_batch_size, target_batch_num_per_read) {
    _tfDefault = tfDefault;
    _globalFreq = globalFreq;
  };

  void readDataset(std::vector<std::string> files, TFIDF_FILE_TYPE type);
  void loadNextBatchSet();
};

}  // namespace thirdai::utils