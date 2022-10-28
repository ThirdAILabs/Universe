#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "BlockInterface.h"
#include <hashing/src/UniversalHash.h>
#include <_types/_uint32_t.h>
#include <dataset/src/batch_processors/TabularMetadataProcessor.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <exception>

namespace thirdai::dataset {

/**
 * @brief Given some metadata about a tabular dataset, assign unique categories
 * to columns and compute pairgrams of the categories.
 *
 * TODO(david): add a TabularBinningStrategy class to try out different methods
 */
class TabularPairGram : public Block {
 public:
  TabularPairGram(TabularMetadataPtr metadata, uint32_t output_range,
                  bool with_pairgrams = true)
      : _metadata(std::move(metadata)),
        _output_range(output_range),
        _with_pairgrams(with_pairgrams) {}

  uint32_t featureDim() const final { return _output_range; };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final { return _metadata->numColumns(); };

  Explanation explainIndex(
      uint32_t index_within_block,
      const std::vector<std::string_view>& columnar_sample) final {
    (void)columnar_sample;
    (void)index_within_block;
    throw std::invalid_argument(
        "Explain feature is not yet implemented in tabular block!");
  }

 protected:
  // TODO(david) We should always include all unigrams but if the number of
  // columns is too large, this processing time becomes slow. One idea is to
  // cap the number of pairgrams at a certain threshold by selecting random
  // pairs of columns to pairgram together.
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    std::vector<uint32_t> unigram_hashes;
    for (uint32_t col = 0; col < input_row.size(); col++) {
      std::string str_val(input_row[col]);
      switch (_metadata->colType(col)) {
        case TabularDataType::Numeric: {
          std::exception_ptr err;
          uint32_t unigram = _metadata->getNumericHashValue(col, str_val, err);
          if (err) {
            return err;
          }
          unigram_hashes.push_back(unigram);
          break;
        }
        case TabularDataType::Categorical: {
          uint32_t unigram = _metadata->getStringHashValue(str_val, col);
          unigram_hashes.push_back(unigram);
          break;
        }
        case TabularDataType::Label:
        case TabularDataType::Ignore:
          break;
      }
    }

    std::vector<uint32_t> hashes_to_use;
    if (_with_pairgrams) {
      hashes_to_use = TextEncodingUtils::computeRawPairgramsFromUnigrams(
          unigram_hashes, _output_range);
    } else {
      std::for_each(unigram_hashes.begin(), unigram_hashes.end(),
                    [this](uint32_t& unigram_hash) {
                      unigram_hash = unigram_hash % _output_range;
                    });
      hashes_to_use = unigram_hashes;
    }

    TextEncodingUtils::sumRepeatedIndices(hashes_to_use, /* base_value = */ 1.0,
                                          [&](uint32_t pairgram, float value) {
                                            vec.addSparseFeatureToSegment(
                                                pairgram, value);
                                          });

    return nullptr;
  }

 private:
  // Private constructor for cereal
  TabularPairGram() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _metadata, _output_range,
            _with_pairgrams);
  }

  TabularMetadataPtr _metadata;
  uint32_t _output_range;
  bool _with_pairgrams;
};

using TabularPairGramPtr = std::shared_ptr<TabularPairGram>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::TabularPairGram)