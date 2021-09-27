#include "StringDataset.h"

namespace thirdai::utils {
StringDataset::StringDataset(std::string filename,
                             STRING_TYPE load_type, uint64_t target_batch_size,
                             uint64_t target_batch_num_per_load)
    : Dataset(target_batch_size, target_batch_num_per_load) {
  // The sentence loaders have not been fully implemented yet
  switch (load_type) {
    case STRING_TYPE::SENTENCE:
      _loader = new SentenceLoader(filename);
      break;
    default:
      std::cerr << "The chosen loader has not been implemented. Defaulting "
                   "to sentence loader."
                << std::endl;
      _loader = new SentenceLoader(filename);
      break;
  }
  _tri_gram_vectorizer
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
    trigram->vectorize(strings_to_be_vectorized[vec_i], _indices[vec_i],
                           _values[vec_i]);
    _batches[batch_i]._lens[batch_vec_i] = _indices[vec_i].size();
    // This prevents us from having to malloc and delete arrays each time.
    // This works because vectors are guaranteed to store its contents in
    // contiguous memory.
    _batches[batch_i]._indices[batch_vec_i] = &(_indices[vec_i][0]); // can use _indices[vec_i].data()
    _batches[batch_i]._values[batch_vec_i] = &(_values[vec_i][0]);
  }
}
}  // namespace thirdai::utils