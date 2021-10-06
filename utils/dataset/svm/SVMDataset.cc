#include "SVMDataset.h"

namespace thirdai::utils {
SVMDataset::SVMDataset(const std::string& filename, uint64_t target_batch_size,
                       uint64_t target_batch_num_per_load)
    : Dataset(target_batch_size, target_batch_num_per_load), _file(filename) {
  if (_file.bad() || _file.fail() || !_file.good() || !_file.is_open()) {
    throw std::runtime_error("Unable to open file '" + filename + "'");
  }
  _num_batches = 0;
  _times_previously_loaded = 0;
  _num_vecs_total = 0;
}

void SVMDataset::loadNextBatchSet() {
  readDataset();
  _num_batches =
      (_num_vecs_in_batch + _target_batch_size - 1) / _target_batch_size;
  // Option 1. At each subsequent load, delete the previous batch array.
  // Option 2. Keep the previous array and just modify it as necessary.
  // Keeping option 2.
  if (_times_previously_loaded > 0) {
    if (_num_batches > 0) {
      uint32_t size_of_last_batch_in_current_load = std::min(
          _target_batch_size,
          _num_vecs_in_batch - (_num_batches - 1) * _target_batch_size);
      _batches[_num_batches - 1] =
          Batch(size_of_last_batch_in_current_load, BATCH_TYPE::SPARSE,
                LABEL_TYPE::LABELED, ID_TYPE::SEQUENTIAL, 0);
      for (uint32_t batch_in_load = 0; batch_in_load < _num_batches;
           batch_in_load++) {
        _batches[batch_in_load]._starting_id = _num_vecs_total;
        _num_vecs_total += _batches[batch_in_load]._batch_size;
      }
    }
  } else {
    _batches = new Batch[_num_batches];

    for (uint64_t i = 0; i < _num_batches; i++) {
      uint32_t batch_size = std::min(
          _target_batch_size, _num_vecs_in_batch - i * _target_batch_size);
      _batches[i] = Batch(batch_size, BATCH_TYPE::SPARSE, LABEL_TYPE::LABELED,
                          ID_TYPE::SEQUENTIAL, 0);
      _batches[i]._starting_id = _num_vecs_total;
      _num_vecs_total += batch_size;
    }
  }

  createBatches();
  _times_previously_loaded++;
}

void SVMDataset::readDataset() {
  _num_vecs_in_batch = 0;
  std::string line;
  _label_markers.clear();
  _labels.clear();
  _markers.clear();
  _indices.clear();
  _values.clear();
  while (
      ((_target_batch_num_per_load > 0 &&
        _num_vecs_in_batch < _target_batch_num_per_load * _target_batch_size) ||
       _target_batch_num_per_load == 0) &&
      std::getline(_file, line)) {
    std::stringstream stream(line);

    _label_markers.push_back(_labels.size());

    std::string labelstr;
    stream >> labelstr;
    size_t pos;
    while ((pos = labelstr.find(',')) != std::string::npos) {
      _labels.push_back(atoi(labelstr.substr(0, pos).c_str()));
      labelstr = labelstr.substr(pos + 1);
    }
    _labels.push_back(atoi(labelstr.c_str()));

    _markers.push_back(_indices.size());
    std::string nonzero;
    while (stream >> nonzero) {
      pos = nonzero.find(':');
      _indices.push_back(atoi(nonzero.substr(0, pos).c_str()));
      _values.push_back(atof(nonzero.substr(pos + 1).c_str()));
    }
    _num_vecs_in_batch++;
  }

  _label_markers.push_back(_labels.size());
  _markers.push_back(_indices.size());
}

void SVMDataset::createBatches() {
  uint32_t* indices_ptr = _indices.data();
  float* values_ptr = _values.data();
  uint32_t* labels_ptr = _labels.data();

  for (uint64_t batch = 0; batch < _num_batches; batch++) {
    for (uint64_t n = 0; n < _batches[batch]._batch_size; n++) {
      uint64_t indx = batch * _target_batch_size + n;
      _batches[batch]._indices[n] = indices_ptr + _markers[indx];
      _batches[batch]._values[n] = values_ptr + _markers[indx];
      _batches[batch]._lens[n] = _markers[indx + 1] - _markers[indx];

      _batches[batch]._labels[n] = labels_ptr + _label_markers[indx];
      _batches[batch]._label_lens[n] =
          _label_markers[indx + 1] - _label_markers[indx];
    }
  }
}
}  // namespace thirdai::utils
