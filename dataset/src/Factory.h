#pragma once

#include <fstream>

namespace thirdai::dataset {

template <typename BATCH_T>
class Factory {
 public:
  virtual BATCH_T parse(std::ifstream&, uint32_t, uint64_t) = 0;
};

}  // namespace thirdai::dataset