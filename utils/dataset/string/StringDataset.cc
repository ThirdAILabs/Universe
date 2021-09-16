#include "StringDataset.h"

/** 
 * TODO (benitoThree): add mock vectorizer and loader
 * 
 * If we were to test this routine, we would call the loadNextBatchSet() method and check _batches and _num_batches
 * _batches specs:
 *  - Number of elements >= _num_batches
 *  - For each batch:
 *    - the right batch size based on target_batch_size.
 *    - batch size corresponds with size of indices, values, and lens
 *    - lengths of indices and values correspond with lens
 *    - check dim, indices and values against the ground truth produced by the vectorizer
 * _num_batches specs:
 *  - Has to be the right number based on target_batch_num_per_load
 * 
 * The mock vectorizer and loader would have to meet some input and output specifications.
 * loader specs:
 *  - The loader outputs strings that only contain lower-case letters, numbers, or space.
 * vectorizer specs:
 *  - Takes in a string and complains if it has anything other than lower-case letters, numbers or space.
 *  - Each string has to produce a different vector so we can easily check whether the batches are duplicates or not.
 */ 

/**
 * TODO(geordie): 
 * Figure out a new StringDataset that incorporates TFIDF and the idea that we will use all three of trigrams, unigrams and bigrams
 * First I want to figure out the loadNextBatchSet workflow?
 * How am I going to slip in TFIDF and the other grams?
 * TFIDF is all about values, so ..?
 * StringDataset, if TFIDF, might need to know whether this is loaded from a serialized object or if the first pass runs at the start.
 * ^But we probably just want an instantiated GlobalFreq to be passed into StringDataset
 * The vectorizers should also take the GlobalFreq
 * GlobalFreq can also be initialized to just 1, so that value would just be the count
 * Or if GlobalFreq is actually not just 1, then the value could be like, count * IDF
 * So I think I figured out how to do TFIDF
 * As for using all three of the token types, 
 * 1. Remove the TOKEN_TYPE enum
 * 2. Change vectorizer to not have to overwrite the vectors. Instead, they have to assume 
 * that they have to add stuff after what is already there and cannot overwrite anything.
 * 3. Run vectorizer one after another, referring to the same indices and values.
 */



namespace thirdai::utils {
StringDataset::StringDataset(std::string filename, TOKEN_TYPE token_type,
                             LOAD_TYPE load_type, uint64_t target_batch_size,
                             uint64_t target_batch_num_per_load)
    : Dataset(target_batch_size, target_batch_num_per_load) {
  // The sentence loaders have not been fully implemented yet
  switch (load_type) {
    case LOAD_TYPE::SENTENCE:
      _loader = new SentenceLoader(filename);
      break;
    default:
      std::cerr << "The chosen loader has not been implemented. Defaulting "
                   "to sentence loader."
                << std::endl;
      _loader = new SentenceLoader(filename);
      break;
  }

  // Only the character trigram vectorizer has been fully implemented for now.
  switch (token_type) {
    case TOKEN_TYPE::CHAR_TRIGRAM:
      _vectorizer = new TriGramVectorizer();
      break;
    default:
      std::cerr << "The chosen tokenizer has not been implemented. "
                   "Defaulting to character trigram."
                << std::endl;
      _vectorizer = new TriGramVectorizer();
      break;
  };
  _dim = _vectorizer->getDimension();
  _initialized = false;
};

void StringDataset::loadNextBatchSet() {
  // Get rid of the previous batch set.
  if (_initialized) {
    for (size_t i = 0; i < _num_batches; i++) {
      delete &(_batches[i]);
    }
    delete[] _batches;
  }

  // Figure out the number of vectors to load.
  // If _target_batch_num_per_load = 0, then load all vectors from the file.
  // Use a vector because we don't know how many strings will be loaded
  // from the file if _target_batch_num_per_load = 0
  std::vector<std::string> strings_to_be_vectorized;
  uint64_t target_vec_num_per_load;
  size_t vec_count = 0;
  if (_target_batch_num_per_load == 0) {
    strings_to_be_vectorized.reserve(_target_batch_size);
    std::string str_buf;
    while (_loader->loadNextString(str_buf)) {
      strings_to_be_vectorized.push_back(str_buf);
    }
    vec_count = strings_to_be_vectorized.size();
    target_vec_num_per_load = vec_count;
  } else {
    target_vec_num_per_load = _target_batch_num_per_load * _target_batch_size;
    strings_to_be_vectorized.reserve(target_vec_num_per_load);
    while (_loader->loadNextString(strings_to_be_vectorized[vec_count]) &&
           vec_count < target_vec_num_per_load) {
      vec_count++;
    }
  }

  // Only initialize indices and values once.
  // They can be reused the next time a batch set is loaded to save time on
  // deallocating and reallocating memory.
  if (!_initialized) {
    _indices = new std::vector<uint32_t>[target_vec_num_per_load];
    _values = new std::vector<float>[target_vec_num_per_load];
    _initialized = true;
  }

  // Must do to comply with Dataset interface specifications.
  _num_batches = (vec_count + _target_batch_size - 1) / _target_batch_size;

  // Mallocing because Batch doesn't have a default constructor.
  _batches = new Batch[_num_batches];
// Initialize each batch.
#pragma omp parallel for default(none) shared(vec_count)
  for (size_t batch_i = 0; batch_i < _num_batches; batch_i++) {
    uint64_t batch_size = vec_count - batch_i * _target_batch_size;
    _batches[batch_i] =
        Batch(batch_size, BATCH_TYPE::SPARSE, LABEL_TYPE::UNLABELED, _dim);
  }

// Vectorize each string and fill out respective batches.
#pragma omp parallel for default(none) \
    shared(vec_count, strings_to_be_vectorized)
  for (size_t vec_i = 0; vec_i < vec_count; vec_i++) {
    size_t batch_i = vec_i / _target_batch_size;
    size_t batch_vec_i = vec_i - (batch_i * _target_batch_size);
    _vectorizer->vectorize(strings_to_be_vectorized[vec_i], _indices[vec_i],
                           _values[vec_i]);
    _batches[batch_i]._lens[batch_vec_i] = _indices[vec_i].size();
    // This prevents us from having to malloc and delete arrays each time.
    // This works because vectors are guaranteed to store its contents in
    // contiguous memory.
    _batches[batch_i]._indices[batch_vec_i] = &(_indices[vec_i][0]);
    _batches[batch_i]._values[batch_vec_i] = &(_values[vec_i][0]);
  }
}
}  // namespace thirdai::utils