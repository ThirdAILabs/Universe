#include "HashTable.h"
#include <atomic>
#include <random>
#include <unordered_set>
#include <vector>

namespace thirdai::utils {

template <typename LABEL_T, bool USE_RESERVOIR>
class VectorHashTable final : public HashTable<LABEL_T> {
 public:
  // We use SFINAE to make the first constructor resolve when USE_RESERVOIR
  // is false and the other constructor resolve when USE_RESERVOIR is true:
  // https://stackoverflow.com/questions/59703327/overloading-based-on-bool-template-parameter
  // This is a slight bit of magic, I honestly don't completely understand it
  // but it does work quite nicely.
  template <bool b = USE_RESERVOIR,
            std::enable_if_t<!b>* = nullptr>  // when USE_RESERVOIR is false
  VectorHashTable(uint32_t num_tables, uint64_t table_range)
      : _num_tables(num_tables),
        _table_range(table_range),
        _buckets(num_tables * table_range),
        _generated_rand_nums(0),
        _num_elements_tried_insert_into_bucket(0),
        _max_reservoir_size(0) {}

  template <bool b = USE_RESERVOIR,
            std::enable_if_t<b>* = nullptr>  // when USE_RESERVOIR is true
  VectorHashTable(uint32_t num_tables, uint64_t max_reservoir_size,
                  uint64_t table_range, uint64_t seed = time(nullptr),
                  uint64_t max_rand = HashTable<LABEL_T>::DEFAULT_MAX_RAND)
      : _num_tables(num_tables),
        _table_range(table_range),
        _buckets(num_tables * table_range),
        _generated_rand_nums(max_rand),
        _num_elements_tried_insert_into_bucket(num_tables * table_range),
        _max_reservoir_size(max_reservoir_size) {
    std::mt19937 generator(seed);
    for (uint64_t i = 0; i < max_rand; i++) {
      _generated_rand_nums[i] = generator();
    }
  }

  void insert(uint64_t n, LABEL_T const* labels,
              uint32_t const* hashes) override;

  void insertSequential(uint64_t n, LABEL_T start,
                        uint32_t const* hashes) override;

  void queryBySet(uint32_t const* hashes,
                  std::unordered_set<LABEL_T>& store) const override;

  void queryByCount(uint32_t const* hashes,
                    std::vector<uint32_t>& counts) const override;

  void queryByVector(uint32_t const* hashes,
                     std::vector<LABEL_T>& results) const override;

  void clearTables() override;

  uint32_t numTables() const override { return _num_tables; };

  uint64_t tableRange() const override { return _table_range; };

  /** Sorts the contents of each bucket */
  void sortBuckets();

 private:
  /** Insert a label into a hashtable, including reservoir sampling if enabled
   */
  inline void insertIntoTable(LABEL_T label, uint32_t hash, uint32_t table);

  constexpr uint64_t getBucketIndex(uint64_t table, uint64_t hash) const {
    return table * tableRange() + hash;
  }

  const uint32_t _num_tables;
  const uint64_t _table_range;
  std::vector<std::vector<LABEL_T>> _buckets;

  // These will be 0 or length 0 if USE_RESERVOIR is false
  std::vector<uint32_t> _generated_rand_nums;
  std::vector<uint32_t> _num_elements_tried_insert_into_bucket;
  const uint64_t _max_reservoir_size;
  std::atomic<uint32_t> counter = 0;
};

}  // namespace thirdai::utils