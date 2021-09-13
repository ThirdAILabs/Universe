#pragma once
#include "Dataset.h"
#include "MurmurHash.h"
#include "TriGramVectorizer.h"
#include "SentenceLoader.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::utils {

enum class TOKEN_TYPE { CHAR_TRIGRAM, WORD_UNIGRAM, WORD_BIGRAM };
enum class STRING_TYPE { SENTENCE, PARAGRAPH, DOCUMENT };

class StringDataset : public Dataset {
 public:

  StringDataset(std::string filename, 
                TOKEN_TYPE token_type, 
                STRING_TYPE vector_type,
                uint64_t target_batch_size,
                uint64_t target_batch_num_per_load)
      : Dataset(target_batch_size, target_batch_num_per_load) {

    // The sentence loaders have not been fully implemented yet
    switch (vector_type) {
      case STRING_TYPE::SENTENCE:
        _loader = SentenceLoader(filename);
        break;
      default:
        std::cerr << "The chosen loader has not been implemented. Defaulting "
                     "to sentence loader."
                  << std::endl;
        _loader = SentenceLoader(filename);
        break;
    }

    // Only the character trigram vectorizer has been fully implemented for now.
    switch (token_type) {
      case TOKEN_TYPE::CHAR_TRIGRAM:
        _vectorizer = TriGramVectorizer();
        break;
      default:
        std::cerr << "The chosen tokenizer has not been implemented. "
                     "Defaulting to character trigram."
                  << std::endl;
        _vectorizer = TriGramVectorizer();
        break;
    };
    _dim = _vectorizer.getDimension();
    _initialized = false;
  };

  virtual void loadNextBatchSet() {
    // Get rid of the previous batch set.
    if (_initialized) {
      for (size_t i = 0; i < _num_batches; i++) {
        delete &(_batches[i]);
      }
      free((void *) _batches);
    }

    // Figure out the number of vectors to load. 
    // If _target_batch_num_per_load = 0, then load all vectors from the file.
    // Use a vector because we don't know how many strings will be loaded 
    // from the file if _target_batch_num_per_load = 0
    std::vector<std::string> strings_to_be_vectorized; 
    uint64_t target_vec_num_per_load;
    size_t vec_count = 0;
    if (_target_batch_num_per_load == 0) {
      strings_to_be_vectorized = std::vector<std::string>(_target_batch_size);
      std::string str_buf;
      while (_loader.loadNextString(str_buf)) {
        strings_to_be_vectorized.push_back(str_buf);
      }
      vec_count = strings_to_be_vectorized.size();
      target_vec_num_per_load = vec_count;
    } else {
      target_vec_num_per_load = _target_batch_num_per_load * _target_batch_size;
      strings_to_be_vectorized = std::vector<std::string>(target_vec_num_per_load);
      while (_loader.loadNextString(strings_to_be_vectorized[vec_count]) && vec_count < target_vec_num_per_load) {
        vec_count++;
      }
    }
    
    // Only initialize indices and values once. 
    // They can be reused the next time a batch set is loaded to save time on deallocating and reallocating memory.
    if (!_initialized) {  
      _indices = new std::vector<uint32_t>[target_vec_num_per_load];
      _values = new std::vector<float>[target_vec_num_per_load];
      _initialized = true;
    }

    // Must do to comply with Dataset interface specifications.
    _num_batches = (vec_count + _target_batch_size - 1) / _target_batch_size;

    // Mallocing because Batch doesn't have a default constructor.
    _batches = (Batch *) malloc(_num_batches * sizeof(Batch)); 
    // Initialize each batch.
    #pragma omp parallel for
    for (size_t batch_i = 0; batch_i < _num_batches; batch_i++) {
      uint64_t batch_size = vec_count - batch_i * _target_batch_size;
      _batches[batch_i] = Batch(batch_size, BATCH_TYPE::SPARSE, LABEL_TYPE::UNLABELED, _dim);
    }

    // Vectorize each string and fill out respective batches.
    #pragma omp parallel for
    for (size_t vec_i = 0; vec_i < vec_count; vec_i++) {
      size_t batch_i = vec_i / _target_batch_size;
      size_t batch_vec_i = vec_i - (batch_i * _target_batch_size);
      _vectorizer.vectorize(strings_to_be_vectorized[vec_i], _indices[vec_i], _values[vec_i]);
      _batches[batch_i]._lens[batch_vec_i] = _indices[vec_i].size();
      // This prevents us from having to malloc and delete arrays each time.
      // This works because vectors are guaranteed to store its contents in contiguous memory.
      _batches[batch_i]._indices[batch_vec_i] = &(_indices[vec_i][0]);
      _batches[batch_i]._values[batch_vec_i] = &(_values[vec_i][0]);
    }
  }

 private:
  std::vector<uint32_t>* _indices;
  std::vector<float>* _values;
  StringVectorizer _vectorizer;
  StringLoader _loader;
  bool _initialized;
  uint32_t _dim;
};
}  // namespace thirdai::utils