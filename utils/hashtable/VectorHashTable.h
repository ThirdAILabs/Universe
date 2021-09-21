#include "HashTable.h"
#include <unordered_set>
#include <vector>

namespace thirdai::utils {

template <typename Label_t>
class VectorHashTable : public HashTable<Label_t> {
 private:
  uint32_t _num_tables;
  uint64_t _table_range;
  std::vector<std::vector<Label_t>> tables;
  constexpr uint64_t getBucketIndex(uint64_t table, uint64_t hash) const {
    return table * tableRange() + hash;
  }

 public:
  VectorHashTable(uint32_t num_tables, uint64_t table_range);

  void insert(uint64_t n, Label_t const* labels,
              uint32_t const* hashes) override;

  void insertSequential(uint64_t n, Label_t start,
                        uint32_t const* hashes) override;

  void queryBySet(uint32_t const* hashes,
                  std::unordered_set<Label_t>& store) const override;

  void queryByCount(uint32_t const* hashes,
                    std::vector<uint32_t>& counts) const override;

  void queryByVector(uint32_t const* hashes,
                     std::vector<Label_t>& results) const override;

  void clearTables() override;

  uint32_t numTables() const override;

  uint64_t tableRange() const override;

  /** Sorts the contents of each bucket */
  void sortBuckets();
};

}  // namespace thirdai::utils