#pragma once
#include <string>
#include <iostream>
#include <unordered_map>
#include <vector>
#include "Dataset.h"
#include "NGramTokenizer.h"
#include "MurmurHash.h"


namespace thirdai::utils {

  enum class STRING_TOKEN_TYPE { CHAR_TRIGRAM, WORD_UNIGRAM, WORD_BIGRAM };
  enum class STRING_VECTOR_TYPE { SENTENCE, PARAGRAPH, DOCUMENT };

  class StringDataset: public Dataset {

    public:

    // n-gram constructor
    StringDataset(std::string filename, uint64_t target_batch_size, uint64_t target_batch_num_per_load, STRING_TOKEN_TYPE token_type, STRING_VECTOR_TYPE vector_type, uint32_t n)
    : Dataset(target_batch_size, target_batch_num_per_load), _file(filename), _token_type(token_type), _vector_type(vector_type) {
      _hashes = new std::unordered_map<uint32_t, float>[_target_batch_num_per_load];
      _indices = new std::vector<uint32_t>[_target_batch_num_per_load];
      _values = new std::vector<float>[_target_batch_num_per_load];
      switch(token_type) {
        case STRING_TOKEN_TYPE::CHAR_TRIGRAM:
          _tokenizer = NGramTokenizer(3);
          break;
        default:
          std::cerr << "The chosen tokenizer has not been implemented. Defaulting to character trigram" << std::endl;
          _tokenizer = NGramTokenizer(3);
          break;
      };
    };

    void cleanUpLineBuffer() {
      // TODO
    }

    bool notSentenceDelimiter(char c, std::string& str) {
      return c != '.' && c != '?' && c != '!';
    }

    bool loadNextSentence(std::string& sentence_buffer) {
      while (_lb_idx == _line_buffer.length()) {
        if (!std::getline(_file, _line_buffer)) {
          return false;
        } // Need to check whether EOF?
        _lb_idx = 0;
        cleanUpLineBuffer();
      }
      size_t start_lb_idx = _lb_idx;
      bool not_sentence_delimiter = true;
      for (_lb_idx; _lb_idx < _line_buffer.length() && not_sentence_delimiter; _lb_idx++) {
        not_sentence_delimiter = notSentenceDelimiter(_line_buffer[_lb_idx], sentence_buffer);
      }
      sentence_buffer = _line_buffer.substr(start_lb_idx, _lb_idx);
      if (!not_sentence_delimiter) {
        _lb_idx++;
      }
      return true;
    }

    virtual void loadNextBatchSet() {
      uint64_t batch_count = 0;
      std::string strings_to_be_vectorized[_target_batch_num_per_load];
      while (batch_count < _target_batch_num_per_load) {
        bool loaded = loadNextSentence(strings_to_be_vectorized[batch_count]);
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
    std::ifstream _file;
    std::string _line_buffer;
    size_t _lb_idx;
    STRING_TOKEN_TYPE _token_type;
    STRING_VECTOR_TYPE _vector_type;
    std::unordered_map<uint32_t, float> *_hashes;
    std::vector<uint32_t> *_indices;
    std::vector<float> *_values;
    StringTokenizer _tokenizer;

    void init_tokenization(const std::string& str, std::unordered_map<uint32_t, float>& hashes, std::vector<uint32_t>& indices, std::vector<float>& values, uint32_t n) {
      
    }

    // TODO: Benchmark and test
    // TODO: Stemming
    // TODO: Synthesize vectors.
    // TODO: % by dimension?
  };
}