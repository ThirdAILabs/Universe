#pragma once

#include <vector>

namespace thirdai::search {

class IdMap {
 public:
  virtual std::vector<uint64_t> get(uint64_t key) const = 0;

  virtual bool contains(uint64_t key) const = 0;

  virtual void put(uint64_t key, std::vector<uint64_t> value) = 0;

  virtual void append(uint64_t key, uint64_t value) = 0;

  virtual void del(uint64_t key) = 0;

  virtual void save(const std::string& path) const = 0;

  virtual ~IdMap() = default;
};

}  // namespace thirdai::search