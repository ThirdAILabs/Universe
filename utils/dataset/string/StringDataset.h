#pragma once
#include "../Dataset.h"
#include "loaders/SentenceLoader.h"
#include "vectorizers/TriGramVectorizer.h"
//#include "GlobalFreq.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::utils {

enum class FRAGMENT_TYPE { SENTENCE, PARAGRAPH, DOCUMENT };

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
  TriGramVectorizer _tri_gram_vectorizer;
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