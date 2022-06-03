#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace thirdai::dataset {

class Loader {
 public:
  virtual std::optional<std::vector<std::string>> nextBatch(uint32_t batch_size) = 0;

  virtual std::vector<std::string_view> parse(const std::string& line) = 0; 

  virtual void initialize() = 0;
};

} // namespace thirdai::dataset