#pragma once

#include "BlockInterface.h"
#include <dataset/src/batch_processors/TabularMetadataProcessor.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <exception>

namespace thirdai::dataset {
/**
 * Given some metadata about a tabular dataset, assign unique categories to
 * columns and compute pairgrams of the categories.
 */
class TabularPairGram : public Block {
 public:
  TabularPairGram(std::shared_ptr<TabularMetadata> metadata,
                  uint32_t output_range)
      : _metadata(metadata), _output_range(output_range) {}

  uint32_t featureDim() const final { return _output_range; };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final {
    // metadata includes the label which this block doesn't need
    return _metadata->numColumns() - 1;
  };

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
          uint32_t unigram = getNumericHashValue(col, str_val, err);
          if (err) {
            return err;
          }
          unigram_hashes.push_back(unigram);
          break;
        }
        case TabularDataType::Categorical: {
          uint32_t unigram = getStringHashValue(str_val, col);
          unigram_hashes.push_back(unigram);
          break;
        }
        case TabularDataType::Label:
          break;
      }
    }

    std::vector<uint32_t> pairgram_hashes =
        TextEncodingUtils::computeRawPairgramsFromUnigrams(unigram_hashes,
                                                           _output_range);

    TextEncodingUtils::sumRepeatedIndices(
        pairgram_hashes, /* base_value = */ 1.0,
        [&](uint32_t pairgram, float value) {
          vec.addSparseFeatureToSegment(pairgram, value);
        });

    return nullptr;
  }

  uint32_t getStringHashValue(const std::string& str_val, uint32_t col) const {
    // to ensure hashes are unique across columns we add salt based on the col
    const char* char_salt = reinterpret_cast<const char*>(&col);
    std::string str_salt(char_salt, 4);
    std::string unique_category = str_val + str_salt;
    return TextEncodingUtils::computeUnigram(unique_category.data(),
                                             unique_category.size());
  }

  uint32_t getNumericHashValue(uint32_t col, const std::string& str_val,
                               std::exception_ptr& exception_ptr) {
    uint32_t bin = getColBin(col, str_val, exception_ptr);
    // to ensure hashes are unique across columns we add salt based on the col
    uint64_t uniqueBin =
        static_cast<uint64_t>(bin) << 32 | static_cast<uint64_t>(col);
    const char* val_to_hash = reinterpret_cast<const char*>(&uniqueBin);
    return TextEncodingUtils::computeUnigram(val_to_hash, /* len = */ 8);
  }

  double getColBinsize(uint32_t col) {
    return (_metadata->colMax(col) - _metadata->colMin(col)) /
           _metadata->numBins(col);
  }

  uint32_t getColBin(uint32_t col, const std::string& str_val,
                     std::exception_ptr& exception_ptr) {
    // map empty values to their own bin
    if (str_val.empty()) {
      return _metadata->numBins(col);
    }
    double value;
    try {
      value = std::stod(str_val);
    } catch (std::invalid_argument& e) {
      exception_ptr = std::make_exception_ptr(std::invalid_argument(
          "Could not process column " + std::to_string(col) +
          " as type 'numeric.' Received value: '" + str_val + ".'"));

      // Since we have set the block exception above, the program will
      // fail once all threads finish. Since we can't throw an exception within
      // a pragma thread, we just have to keep the program running until then.
      // Thus we return some arbitrary value to do that.
      return 0;
    }
    double binsize = getColBinsize(col);
    if (binsize == 0) {
      return 0;
    }
    return static_cast<uint32_t>(
        std::round((value - _metadata->colMin(col)) / binsize));
  }

 private:
  std::shared_ptr<TabularMetadata> _metadata;
  uint32_t _output_range;
};

}  // namespace thirdai::dataset
