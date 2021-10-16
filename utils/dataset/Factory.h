#pragma once

#include <fstream>

namespace thirdai::utils {

template <typename Batch_t>
class Factory {
 public:
  virtual Batch_t parse(std::ifstream&, uint32_t, uint64_t) = 0;
};

}  // namespace thirdai::utils