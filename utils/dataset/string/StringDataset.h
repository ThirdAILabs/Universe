#pragma once
#include "../Dataset.h"
#include "loaders/SentenceLoader.h"
#include "vectorizers/TriGramVectorizer.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::utils {

enum class TOKEN_TYPE { CHAR_TRIGRAM, WORD_UNIGRAM, WORD_BIGRAM };
enum class LOAD_TYPE { SENTENCE, PARAGRAPH, DOCUMENT };

class StringDataset : public Dataset {
 public:
  StringDataset(std::string filename, TOKEN_TYPE token_type,
                LOAD_TYPE load_type, uint64_t target_batch_size,
                uint64_t target_batch_num_per_load);

  virtual void loadNextBatchSet();

 private:
  std::vector<uint32_t>* _indices;
  std::vector<float>* _values;
  StringVectorizer* _vectorizer;
  StringLoader* _loader;
  bool _initialized;
  uint32_t _dim;
};
}  // namespace thirdai::utils