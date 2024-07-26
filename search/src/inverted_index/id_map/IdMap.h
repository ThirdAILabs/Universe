#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace thirdai::search {

class IdMap {
 public:
  /**
   * Returns the values associated with a key. This is used to map the query id
   * to the doc ids it corresponds to.
   */
  virtual std::vector<uint64_t> get(uint64_t key) const = 0;

  /**
   * Adds a new key and set of values to the mapping. This is used to add a new
   * query id and corresponding set of doc ids.
   */
  virtual void put(uint64_t key, const std::vector<uint64_t>& values) = 0;

  /**
   * Deletes a value, returns a list of keys that mapped to only that value.
   * This is used to delete doc ids, and then delete any queries that only
   * mapped to that doc id.
   */
  virtual std::vector<uint64_t> deleteValue(uint64_t value) = 0;

  virtual void save(const std::string& path) const = 0;

  virtual std::string type() const = 0;

  virtual ~IdMap() = default;
};

}  // namespace thirdai::search