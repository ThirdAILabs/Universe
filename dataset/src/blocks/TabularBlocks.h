#pragma once

#include "BlockInterface.h"
#include <dataset/src/bolt_datasets/batch_processors/PairgramHasher.h>
#include <dataset/src/bolt_datasets/batch_processors/TabularMetadataProcessor.h>
#include <unordered_map>

namespace thirdai::dataset {

class TabularPairGram : public Block {
 public:
  TabularPairGram(std::shared_ptr<TabularMetadata> metadata,
                  uint32_t output_range)
      : _metadata(metadata), _output_range(output_range) {}

  uint32_t featureDim() const final { return _output_range; };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final { return _metadata->numColumns(); };

 protected:
  // TODO(david) calculate unigrams and capped pairgrams correctly
  void buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec) final {
    std::vector<uint32_t> unigram_hashes;
    for (uint32_t col = 0; col < input_row.size(); col++) {
      uint32_t unigram;
      switch (_metadata->getType(col)) {
        case TabularDataType::Numeric: {
          // TODO(david) if stof fails
          std::string string_value(input_row[col]);
          float value = std::stof(string_value);
          std::string bin = _metadata->getColBin(col, value);
          unigram_hashes.push_back(calculateUnigram(bin));
          break;
        }
        case TabularDataType::Categorical: {
          unigram_hashes.push_back(calculateUnigram(input_row[col]));
          break;
        }
        case TabularDataType::Label:
          break;
      }
      unigram_hashes.push_back(unigram);
      vec.addSparseFeatureToSegment(unigram % _output_range,
                                    1.0);  // TODO(david) update
    }

    // at the minimum must have unigrams
    // if we have less than 1000 unigrams, add pairgrams of random columns (same
    // random columns for each row) up to a cap of 1000
  }

 private:
  uint32_t calculateUnigram(const std::string_view& item) {
    return hashing::MurmurHash(item.data(), item.size(), HASH_SEED);
  }

  static constexpr uint32_t HASH_SEED = 3829;
  std::shared_ptr<TabularMetadata> _metadata;
  uint32_t _output_range;
};

class TabularLabel : public Block {
 public:
  explicit TabularLabel(std::shared_ptr<TabularMetadata> metadata)
      : _metadata(metadata) {}

  uint32_t featureDim() const final { return _metadata->numClasses(); };

  bool isDense() const final { return true; };

  uint32_t expectedNumColumns() const final { return 1; };

 protected:
  void buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec) final {
    vec.addSparseFeatureToSegment(_metadata->getClassId(input_row), 1.0);
  }

 private:
  std::shared_ptr<TabularMetadata> _metadata;
};

}  // namespace thirdai::dataset
