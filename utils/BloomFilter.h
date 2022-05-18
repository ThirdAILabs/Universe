#include <hashing/src/MurmurHash.h>
#include <iostream>
#include <vector>

namespace thirdai::utils{

template <typename KEY_T>
class BloomFilter{
    private:
        uint64_t _capacity, _fp_rate, _R, _K, count;
        std::vector<uint32_t> _seed_array;

    public:
    /*
        this BloomFilter must be able to store at least *capacity* elements
        while maintaining no more than *fp_rate* chance of false
        positives
    */
    BloomFilter(uint64_t capacity, uint64_t fp_rate);

    BloomFilter(const BloomFilter& other) = delete;

    BloomFilter& operater=(const BloomFilter& other) = delete;

    ~BloomFilter();

    void insert(KEY_T key);

    bool contains(KEY_T query);

    void clear();

    uint64_t size();
};

}   // namespace thirdai::utils