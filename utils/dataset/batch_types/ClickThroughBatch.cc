#include "ClickThroughBatch.h"
#include <_types/_uint32_t.h>
#include <cstdlib>

namespace thirdai::utils {

ClickThroughBatch::ClickThroughBatch(std::ifstream& file,
                                     uint32_t target_batch_size,
                                     uint32_t start_id,
                                     const BatchOptions& options)
    : _batch_size(0), _start_id(start_id) {
  std::string line;
  while (_batch_size < target_batch_size && std::getline(file, line)) {
    const char* start = line.c_str();
    char* end;

    uint32_t label = std::strtol(start, &end, 10);
    _labels.push_back(label);

    start = end + 1;

    DenseVector vec(options.click_through.dense_features);
    for (uint32_t d = 0; d < options.click_through.dense_features; d++) {
      float feature = std::strtof(start, &end);
      start = end + 1;
      vec.values[d] = feature;
    }
    _dense_features.push_back(std::move(vec));

    std::vector<uint32_t> categorical(
        options.click_through.categorical_features);
    for (uint32_t c = 0; c < options.click_through.categorical_features; c++) {
      uint32_t feature = std::strtol(start, &end, 10);
      start = end + 1;
      categorical[c] = feature;
    }
    _categorical_features.push_back(std::move(categorical));

    _batch_size++;
  }
}

}  // namespace thirdai::utils