#pragma once
#include "../Dataset.h"
#include "loaders/SentenceLoader.h"
#include "vectorizers/TriGramVectorizer.h"
#include "GlobalFreq.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::utils {

enum class FRAGMENT_TYPE { SENTENCE, PARAGRAPH, DOCUMENT };
// enum class VECTOR_TYPE { TFIDF, MURMUR };

/**
 * TODO(geordie):
 * Figure out a new StringDataset that incorporates TFIDF and the idea that we
 * will use all three of trigrams, unigrams and bigrams First I want to figure
 * out the loadNextBatchSet workflow? How am I going to slip in TFIDF and the
 * other grams? TFIDF is all about values, so ..? StringDataset, if TFIDF, might
 * need to know whether this is loaded from a serialized object or if the first
 * pass runs at the start. ^But we probably just want an instantiated GlobalFreq
 * to be passed into StringDataset The vectorizers should also take the
 * GlobalFreq GlobalFreq can also be initialized to just 1, so that value would
 * just be the count Or if GlobalFreq is actually not just 1, then the value
 * could be like, count * IDF So I think I figured out how to do TFIDF As for
 * using all three of the token types,
 * 1. Remove the TOKEN_TYPE enum
 * 2. Change vectorizer to not have to overwrite the vectors. Instead, they have
 * to assume that they have to add stuff after what is already there and cannot
 * overwrite anything.
 * 3. Run vectorizer one after another, referring to the same indices and
 * values.
 */

class StringDataset : public Dataset {
 public:
  StringDataset(FRAGMENT_TYPE load_type, uint64_t target_batch_size,
                uint64_t target_batch_num_per_load);

  void loadNextBatchSet() override;

  /**
   * Queues a dataset file.
   * Files will be automatically read in order.
   * Does not check for duplicates.
   */
  void addFileToQueue(std::string filename);

 private:
  std::vector<uint32_t>* _indices;
  std::vector<float>* _values;
  TriGramVectorizer* _tri_gram_vectorizer;
  StringLoader* _loader;
  bool _first_load;
  uint32_t _tri_gram_dim;
  uint32_t _dim = 0;

  /**
   * Helper function for initializing _values, _indices, and _batches
   * with the right configurations (e.g. size, dimension, etc) so that
   * they are ready to be filled by vectorizeAndCreateBatches().
   */
  void initializeValuesIndicesBatches(size_t& vec_count);

  /**
   * Helper function for vectorizing strings and filling out batches.
   */
  void vectorizeAndCreateBatches(
      size_t& vec_count, std::vector<std::string>& strings_to_be_vectorized);
};
}  // namespace thirdai::utils