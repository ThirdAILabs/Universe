#include "SparseBatch.h"
#include <cstdlib>

namespace thirdai::utils {

SparseBatch::SparseBatch(std::ifstream& file, uint32_t target_batch_size,
                         uint64_t start_id, const BatchOptions& /* options */)
    : _batch_size(0), _start_id(start_id) {
  std::string line;
  while (_batch_size < target_batch_size && std::getline(file, line)) {
    const char* start = line.c_str();
    char* end;
    std::vector<uint32_t> labels;
    do {
      uint32_t label = std::strtoul(start, &end, 10);
      labels.push_back(label);
      start = end;
    } while ((*start++) == ',');
    this->_labels.push_back(std::move(labels));

    std::vector<std::pair<uint32_t, float>> nonzeros;
    do {
      uint32_t index = std::strtoul(start, &end, 10);
      start = end + 1;
      float value = std::strtof(start, &end);
      nonzeros.push_back({index, value});
      start = end;
    } while ((*start++) == ' ');

    SparseVector v(nonzeros.size());
    uint32_t cnt = 0;
    for (const auto& x : nonzeros) {
      v.indices[cnt] = x.first;
      v.values[cnt] = x.second;
      cnt++;
    }

    _vectors.push_back(std::move(v));
    _batch_size++;
  }
}

}  // namespace thirdai::utils