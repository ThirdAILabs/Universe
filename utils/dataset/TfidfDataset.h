#include "Dataset.h"
#include "GlobalFreq.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>

namespace thirdai::utils {

#define MAX_LEN 50;

enum class TFIDF_FILE_TYPE{ FILE, SINGLE_SENTENCE, PARAGRAPH, N_SENTENCE };
enum class TOKEN_TYPE { UNI_GRAM, TRI_GRAM };

class TfidfDataset : public Dataset {
 private:
  // TODO: Store a hash function(eg. murmurHash) that maps tokens to token ids
  // (indices), we can also use default natural numbers.
  // NOTE: This is just a placeholder!!!
  std::unordered_map<std::string, int> _tokenIdMap;

  int _idfDefault = 1;  // Specifies whether we should use a default value for
                        // the TF of a word.
  GlobalFreq* _globalFreq;
  std::vector<uint32_t> _indices;
  std::vector<float> _values;
  std::vector<uint32_t> _markers;
  std::vector<uint32_t> _labels;
  std::vector<uint32_t> _label_markers;

  // TODO: Also need a tokenization tool/object for word-unigram and
  // char-trigram;

 public:
  TfidfDataset(uint64_t target_batch_size, uint64_t target_batch_num_per_read,
               GlobalFreq* globalFreq, int idfDefault = 1)
      : Dataset(target_batch_size, target_batch_num_per_read) {
    _idfDefault = idfDefault;
    _globalFreq = globalFreq;
  };
 
  /*
    Want to assume that the strings in the files are cleaned and good
  */
  void readDataset(std::vector<std::string> files, TFIDF_FILE_TYPE fileType, TOKEN_TYPE tokenType = TOKEN_TYPE::UNI_GRAM);
  void loadNextBatchSet();
};

}  // namespace thirdai::utils