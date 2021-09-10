#pragma once
#include <string>
#include <iostream>
#include <unordered_map>
#include <vector>
#include "Dataset.h"
#include "NGramTokenizer.h"
#include "MurmurHash.h"
#include "SentenceLoader.h"


namespace thirdai::utils {

  enum class STRING_TOKEN_TYPE { CHAR_TRIGRAM, WORD_UNIGRAM, WORD_BIGRAM };
  enum class STRING_VECTOR_TYPE { SENTENCE, PARAGRAPH, DOCUMENT };

  class StringDataset: public Dataset {

    public:

    // n-gram constructor
    StringDataset(std::string filename, uint64_t target_batch_size, uint64_t target_batch_num_per_load, STRING_TOKEN_TYPE token_type, STRING_VECTOR_TYPE vector_type, uint32_t n)
    : Dataset(target_batch_size, target_batch_num_per_load), _token_type(token_type), _vector_type(vector_type) {
      _hashes = new std::unordered_map<uint32_t, float>[_target_batch_num_per_load];
      _indices = new std::vector<uint32_t>[_target_batch_num_per_load];
      _values = new std::vector<float>[_target_batch_num_per_load];
      switch(token_type) {
        case STRING_TOKEN_TYPE::CHAR_TRIGRAM:
          _tokenizer = NGramTokenizer(3);
          break;
        default:
          std::cerr << "The chosen tokenizer has not been implemented. Defaulting to character trigram." << std::endl;
          _tokenizer = NGramTokenizer(3);
          break;
      };
      switch (vector_type)
      {
        case STRING_VECTOR_TYPE::SENTENCE:
          _loader = SentenceLoader(filename);
          break;
        default:
          std::cerr << "The chosen loader has not been implemented. Defaulting to sentence loader." << std::endl;
          _loader = SentenceLoader(filename);
          break;
      }
    };

    void cleanUpLineBuffer() {
      // TODO
    }

    bool notSentenceDelimiter(char c, std::string& str) {
      return c != '.' && c != '?' && c != '!';
    }

    virtual void loadNextBatchSet() {
      uint64_t batch_count = 0;
      std::string strings_to_be_vectorized[_target_batch_num_per_load];
      while (batch_count < _target_batch_num_per_load) {
        bool loaded = _loader.loadNextString(strings_to_be_vectorized[batch_count]);
        if (!loaded) {
          break;
        }
        batch_count++;
      }
      #pragma omp parallel for
      for (uint64_t i = 0; i < batch_count; i++) {
        _tokenizer.tokenize(strings_to_be_vectorized[i], _hashes[i], _indices[i], _values[i]);
      }
    }

    private:
    STRING_TOKEN_TYPE _token_type;
    STRING_VECTOR_TYPE _vector_type;
    std::unordered_map<uint32_t, float> *_hashes;
    std::vector<uint32_t> *_indices;
    std::vector<float> *_values;
    StringTokenizer _tokenizer;
    StringLoader _loader;

    // TODO: Benchmark and test
    // TODO: Stemming
    // TODO: Synthesize vectors.
    // TODO: % by dimension?
  };
}