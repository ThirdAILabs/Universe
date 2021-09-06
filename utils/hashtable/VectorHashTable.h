#include "HashTable.h"
#include <unordered_set>
#include <vector>

namespace thirdai::utils {

template <typename Label_t>
class VectorHashTable : public HashTable<Label_t> {
 public:
  VectorHashTable(uint64_t num_tables, uint64_t table_range);

  void insert(uint64_t n, Label_t const* labels,
              uint32_t const* hashes) override;

  void insertSequential(uint64_t n, Label_t start,
                        uint32_t const* hashes) override;

  void queryBySet(uint32_t const* hashes,
                  const std::unordered_set<Label_t>& store) override;

  void queryByCount(uint32_t const* hashes,
                    const std::vector<Label_t>& counts) override;

  void queryByVector(uint32_t const* hashes,
                     const std::vector<Label_t>& results) override;

  void clearTables() override;

  uint64_t numTables() override;

  uint64_t tableRange() override;

  void sortBuckets();

 private:
  constexpr uint64_t getBucket(uint64_t table, uint64_t hash) {
    return tables[table * tableRange() + hash];
  }

  std::vector<std::vector<Label_t>> tables;

  uint64_t _num_tables, _table_range;
};

}  // namespace thirdai::utils