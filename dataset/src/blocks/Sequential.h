#include "BlockInterface.h"
#include <hashing/src/MurmurHash.h>
#include <dataset/src/utils/CircularBuffer.h>
#include <algorithm>
#include <charconv>
#include <unordered_map>

namespace thirdai::dataset {

class SequentialBlock : public Block {
 public:
  SequentialBlock(uint32_t user_col, uint32_t item_col, uint32_t last_n,
                  uint32_t dim)
      : _user_col(user_col), _item_col(item_col), _last_n(last_n), _dim(dim) {}

  uint32_t featureDim() const final { return _dim; }

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final {
    return std::max(_item_col, _user_col) + 1;
  };

  static constexpr uint32_t HASH_SEED = 341;

 protected:
  void buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec) final {
    std::string user(input_row[_user_col]);
    auto item_str = input_row[_item_col];
    uint32_t item =
        hashing::MurmurHash(item_str.data(), item_str.size(), HASH_SEED);

    // TODO(Geordie): Can we try to name critical section? Can it be made
    // smaller?
#pragma omp critical
    {
      if (_histories[user].size() == _last_n) {
        _histories[user].pop();
      }
      _histories[user].insert(std::move(item));
      for (size_t i = 0; i < _histories[user].size(); i++) {
        vec.addSparseFeatureToSegment(_histories[user].at(i), 1.0);
      }
    }
  }

 private:
  uint32_t _user_col;
  uint32_t _item_col;
  uint32_t _last_n;
  uint32_t _dim;
  // TODO(Geordie): How to parallel?
  std::unordered_map<std::string, CircularBuffer<uint32_t>> _histories;
};

}  // namespace thirdai::dataset