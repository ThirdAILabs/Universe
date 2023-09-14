#pragma once
#include <bolt_vector/src/BoltVector.h>
#include <data/src/ColumnMap.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/featurizers/SvmFeaturizer.h>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

class DyadicFeaturizer final : public Featurizer {
 public:
  explicit DyadicFeaturizer(bool expects_header, size_t n_intervals,
                            size_t context_length, std::string text_column,
                            std::string label_column, char delimiter,
                            char label_delimiter)
      : _expects_header(expects_header),
        _n_intervals(n_intervals),
        _context_length(context_length),
        _text_column(std::move(text_column)),
        _label_column(std::move(label_column)),
        _output_interval_prefix("dyadic_" + _text_column),
        _delimiter(delimiter),
        _label_delimiter(label_delimiter) {}

  bool expectsHeader() const final { return true; }

  void processHeader(const std::string& header) final {
    _column_number_map = dataset::ColumnNumberMap(header, _delimiter);
    if (!_column_number_map->containsColumn(_text_column)) {
      throw std::invalid_argument("Expected Column " + _text_column +
                                  " in the dataset but found none.");
    }
    _num_cols_in_header = _column_number_map->size();
  }

  std::vector<std::vector<BoltVector>> featurize(
      const std::vector<std::string>& rows) final;

  size_t getNumDatasets() final { return _n_intervals; }

  static std::vector<uint32_t> convertStringToUInt32FeatureArray(
      const std::string& str) {
    std::istringstream iss(str);
    std::vector<uint32_t> result((std::istream_iterator<uint32_t>(iss)),
                                 std::istream_iterator<uint32_t>());
    return result;
  }

  std::vector<uint32_t> convertStringToUInt32LabelArray(
      const std::string& str) const {
    std::istringstream iss(str);
    std::vector<uint32_t> result;
    std::string token;

    while (std::getline(iss, token, _label_delimiter)) {
      try {
        uint32_t value = std::stoul(token);  // Convert string to uint32_t
        result.push_back(value);
      } catch (const std::invalid_argument& e) {
        // Handle invalid token, for now, we'll just continue to the next token
        continue;
      }
    }

    return result;
  }

  std::vector<BoltVector> featurizeSingle(
      const std::vector<uint32_t>& tokens) const;

 private:
  bool _expects_header;
  size_t _n_intervals;
  size_t _context_length;

  std::string _text_column;
  std::string _label_column;
  std::string _output_interval_prefix;
  char _delimiter;
  char _label_delimiter;

  std::optional<uint32_t> _num_cols_in_header = std::nullopt;

  std::optional<dataset::ColumnNumberMap> _column_number_map = std::nullopt;
};

using DyadicFeaturizerPtr = std::shared_ptr<DyadicFeaturizer>;
}  // namespace thirdai::dataset