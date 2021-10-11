#include "StringDataset.h"

namespace thirdai::utils {
StringDataset::StringDataset(FRAGMENT_TYPE load_type,
                             uint64_t target_batch_size,
                             uint64_t target_batch_num_per_load)
    : Dataset(target_batch_size, target_batch_num_per_load),
      _char_tri_gram_vectorizer(0, 100000),
      _word_uni_gram_vectorizer(0, 100000, VALUE_TYPE::TF) {
  // The loaders have not been fully implemented yet. Only sentence loader is
  // available for now.
  switch (load_type) {
    case FRAGMENT_TYPE::SENTENCE:
      _loader = new SentenceLoader();
      break;
    default:
      std::cerr << "The chosen loader has not been implemented. Defaulting "
                   "to sentence loader until other loaders are ready."
                << std::endl;
      _loader = new SentenceLoader();
      break;
  }
  _dim += _char_tri_gram_vectorizer.getDimension();
  _word_uni_gram_vectorizer = UnigramVectorizer(_dim, 100000, VALUE_TYPE::TF);
  _dim += _word_uni_gram_vectorizer.getDimension();
  _first_load = true;
}

void StringDataset::addFileToQueue(std::string filename) {
  _loader->addFileToQueue(filename);
}

void StringDataset::processGlobal(std::vector<std::string>& files) {
  // TODO: (Henry) Where do we want to store the seed?
  _global_freq = new GlobalFreq(files, _loader, 42);
  //_word_uni_gram_vectorizer.setGlobalFreq(_global_freq);
}

void StringDataset::loadNextBatchSet() {
  // Load the first set of strings while computing the number of vectors to
  // load. If _target_batch_num_per_load = 0, then load all vectors from the
  // file.
  std::vector<std::string> strings_to_be_vectorized;
  uint64_t vec_count = 0;
  std::string str_buf;
  if (_target_batch_num_per_load > 0) {
    strings_to_be_vectorized.reserve(_target_batch_num_per_load *
                                     _target_batch_size);
  }
  while ((_target_batch_num_per_load == 0 ||
          vec_count < _target_batch_num_per_load * _target_batch_size) &&
         _loader->loadNextString(str_buf)) {
    strings_to_be_vectorized.push_back(str_buf);
    vec_count++;
  }

  _num_batches = (vec_count + _target_batch_size - 1) / _target_batch_size;

  initializeValuesIndicesBatches(vec_count);

  vectorizeAndCreateBatches(vec_count, strings_to_be_vectorized);

  _first_load = false;
}

void StringDataset::initializeValuesIndicesBatches(uint64_t& vec_count) {
  // Only initialize indices, values and batches once.
  // They can be reused the next time a batch set is loaded to save time on
  // deallocating and reallocating memory.
  // Note that the number of vectors in the first load will be greater than or
  // equal to the number of vectors in subsequent loads.
  if (_first_load) {
    _indices = new std::vector<uint32_t>[vec_count];
    _values = new std::vector<float>[vec_count];
    _batches = new Batch[_num_batches];
#pragma omp parallel for default(none) shared(vec_count)
    for (uint64_t batch_i = 0; batch_i < _num_batches; batch_i++) {
      uint64_t batch_size = std::min(_target_batch_size,
                                     vec_count - batch_i * _target_batch_size);
      _batches[batch_i] =
          Batch(batch_size, BATCH_TYPE::SPARSE, LABEL_TYPE::UNLABELED,
                ID_TYPE::SEQUENTIAL, _dim);
    }
  } else {
    // No need to compute anything if _num_batches = 0.
    // Only the last batch in the current load can have a batch size <
    // _target_batch_size, so only this batch may need to be reinitialized in
    // subsequent loads.
    if (_num_batches > 0) {
      uint64_t size_of_last_batch_in_current_load =
          vec_count - (_num_batches - 1) * _target_batch_size;
      if (size_of_last_batch_in_current_load < _target_batch_size) {
        _batches[_num_batches - 1] =
            Batch(size_of_last_batch_in_current_load, BATCH_TYPE::SPARSE,
                  LABEL_TYPE::UNLABELED, ID_TYPE::SEQUENTIAL, _dim);
      }
    }
  }
}

void StringDataset::vectorizeAndCreateBatches(
    uint64_t& vec_count, std::vector<std::string>& strings_to_be_vectorized) {
#pragma omp parallel for default(none) \
    shared(vec_count, strings_to_be_vectorized)
  for (uint64_t vec_i = 0; vec_i < vec_count; vec_i++) {
    _char_tri_gram_vectorizer.vectorize(strings_to_be_vectorized[vec_i],
                                        _indices[vec_i], _values[vec_i]);
    _word_uni_gram_vectorizer.vectorize(strings_to_be_vectorized[vec_i],
                                        _indices[vec_i], _values[vec_i]);

    uint64_t batch_i = vec_i / _target_batch_size;
    uint64_t batch_vec_i = vec_i - (batch_i * _target_batch_size);
    _batches[batch_i]._lens[batch_vec_i] = _indices[vec_i].size();
    // Vectors are guaranteed to store its contents in
    // contiguous memory. This prevents allocating and deleting arrays each
    // time.
    _batches[batch_i]._indices[batch_vec_i] = _indices[vec_i].data();
    _batches[batch_i]._values[batch_vec_i] = _values[vec_i].data();
  }
  for (uint64_t i = 0; i < _num_batches; i++) {
    _batches[i]._starting_id = _batch_set_starting_id + i * _target_batch_size;
  }
}
}  // namespace thirdai::utils
