#pragma once
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <data/src/ColumnMap.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/featurizers/SvmFeaturizer.h>
#include <dataset/src/mach/MachBlock.h>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

class DyadicFeaturizer final : public Featurizer {
 public:
  explicit DyadicFeaturizer(
      bool expects_header, size_t n_intervals, size_t context_length,
      std::string text_column, char delimiter,
      std::optional<std::string> label_column = std::nullopt,
      std::optional<char> label_delimiter = std::nullopt,
      std::optional<mach::MachBlockPtr> mach_label_block = std::nullopt)
      : _expects_header(expects_header),
        _n_intervals(n_intervals),
        _context_length(context_length),
        _text_column(std::move(text_column)),
        _output_interval_prefix("dyadic_" + _text_column),
        _delimiter(delimiter),
        _label_column(std::move(label_column)),
        _label_delimiter(label_delimiter),
        _mach_label_block(std::move(mach_label_block)) {}

  bool expectsHeader() const final { return true; }

  void processHeader(const std::string& header) final {
    _column_number_map = dataset::ColumnNumberMap(header, _delimiter);
    if (!_column_number_map->containsColumn(_text_column)) {
      throw std::invalid_argument("Expected Column " + _text_column +
                                  " in the dataset but found none.");
    }
    _num_cols_in_header = _column_number_map->size();
    auto num_to_name_map = _column_number_map->getColumnNumToColNameMap();
    // for (size_t i = 0; i < num_to_name_map.size(); i++) {
    //   std::cout << "ID: " << i << " Name: " << num_to_name_map[i] <<
    //   std::endl;
    // }
  }

  size_t getNumDatasets() final {
    // no labels present
    if (_label_column == std::nullopt) {
      return _n_intervals;
    }
    // only doc labels present
    if (_mach_label_block == std::nullopt) {
      return _n_intervals + 1;
    }
    // both mach and doc labels are present
    return _n_intervals + 2;
  }

  std::vector<std::vector<BoltVector>> featurize(
      const std::vector<std::string>& rows) override;

  std::vector<std::vector<BoltVector>> featurize(
      const MapInputBatch& map_input_batch);

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

    while (std::getline(iss, token, _label_delimiter.value())) {
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

  size_t n_intervals() const { return _n_intervals; }

  size_t context_length() const { return _context_length; }

  std::string text_column() const { return _text_column; }

  char delimiter() const { return _delimiter; }

  std::optional<std::string> label_column() const { return _label_column; }

  std::optional<char> label_delimiter() const { return _label_delimiter; }

  std::optional<mach::MachBlockPtr> mach_label_block() const {
    return _mach_label_block;
  }

  DyadicFeaturizer makeInferenceFeaturizer() {
    return DyadicFeaturizer(false, _n_intervals, _context_length, _text_column,
                            _delimiter);
  }

  std::vector<uint32_t> getDimensions() final {
    if (_label_column == std::nullopt) {
      return std::vector<uint32_t>(_n_intervals, _context_length);
    }
    if (_mach_label_block == std::nullopt) {
      return std::vector<uint32_t>(_n_intervals + 1, _context_length);
    }
    return std::vector<uint32_t>(_n_intervals + 2, _context_length);
  }

 private:
  DyadicFeaturizer() {}
  std::vector<std::vector<BoltVector>> featurize(
      ColumnarInputBatch& inputs_ref);

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  bool _expects_header;
  size_t _n_intervals;
  size_t _context_length;

  std::string _text_column;
  std::string _output_interval_prefix;
  char _delimiter;
  std::optional<std::string> _label_column;
  std::optional<char> _label_delimiter;

  std::optional<mach::MachBlockPtr> _mach_label_block;

  std::optional<uint32_t> _num_cols_in_header = std::nullopt;

  std::optional<dataset::ColumnNumberMap> _column_number_map = std::nullopt;
};

using DyadicFeaturizerPtr = std::shared_ptr<DyadicFeaturizer>;
}  // namespace thirdai::dataset