#pragma once

#include <dataset/src/blocks/ColumnIdentifier.h>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace thirdai::dataset {

// Single input type aliases
using MapInput = std::unordered_map<std::string, std::string>;
using RowInput = std::vector<std::string_view>;
using LineInput = std::string;

// Batch input type aliases
using MapInputBatch = std::vector<std::unordered_map<std::string, std::string>>;
using LineInputBatch = std::vector<std::string>;

/**
 * An interface that allows the data pipeline to operate on arbitrary
 * representations of a single columnar input sample.
 *
 *
 *
 * SingleInputRef is short for single input reference. Single -
 *
 * An interface for wrappers around a single columnar input sample of an
 * arbitrary format.
 * It is called "single" input ref because the underlying data structure
 * represents a single input sample.
 */
class ColumnarInputSample {
 public:
  // TODO(Geordie): Add methods to parse column into their real types?
  // E.g. "1" -> int(1)

  /**
   * Get
   */
  virtual std::string_view column(const ColumnIdentifier& column) = 0;
  virtual uint32_t size() = 0;
  virtual ~ColumnarInputSample() = default;
};

class MapSampleRef final : public ColumnarInputSample {
 public:
  // NOLINTNEXTLINE Implicit constructor is intentional
  MapSampleRef(const MapInput& columns) : _columns(columns) {}

  std::string_view column(const ColumnIdentifier& column) final {
    if (!_columns.count(column.name())) {
      return {};
    }
    return _columns.at(column.name());
  }

  uint32_t size() final { return _columns.size(); }

 private:
  const MapInput& _columns;
};

class RowSampleRef final : public ColumnarInputSample {
 public:
  // NOLINTNEXTLINE Implicit constructor is intentional
  RowSampleRef(const RowInput& columns) : _columns(columns) {}

  std::string_view column(const ColumnIdentifier& column) final {
    return _columns.at(column.number());
  }

  uint32_t size() final { return _columns.size(); }

 private:
  const RowInput& _columns;
};

class CsvSampleRef final : public ColumnarInputSample {
 public:
  CsvSampleRef(const LineInput& line, char delimiter,
               std::optional<uint32_t> expected_num_cols = std::nullopt)
      : _columns(ProcessorUtils::parseCsvRow(line, delimiter)) {
    if (expected_num_cols && expected_num_cols != _columns->size()) {
      std::stringstream error_ss;
      error_ss << "Expected " << *expected_num_cols
               << " columns in each row of the dataset. Found row with "
               << _columns->size() << " columns:";
      throw std::invalid_argument(error_ss.str());
    }
  }

  std::string_view column(const ColumnIdentifier& column) final {
    return RowSampleRef(*_columns).column(column);
  }

  uint32_t size() final { return _columns->size(); }

 private:
  std::optional<RowInput> _columns;
};

class ColumnarInputBatch {
 public:
  virtual ColumnarInputSample& sample(uint32_t index) = 0;
  virtual uint32_t size() = 0;
  virtual ~ColumnarInputBatch() = default;
};

class MapBatchRef final : public ColumnarInputBatch {
 public:
  // NOLINTNEXTLINE
  MapBatchRef(const MapInputBatch& batch) : _batch(batch) {
    _ref_batch.reserve(_batch.size());
    for (const auto& input : _batch) {
      _ref_batch.emplace_back(MapSampleRef(input));
    }
  }

  ColumnarInputSample& sample(uint32_t index) final {
    return _ref_batch.at(index);
  }

  uint32_t size() final { return _batch.size(); }

 private:
  const MapInputBatch& _batch;
  std::vector<MapSampleRef> _ref_batch;
};

class CsvBatchRef final : public ColumnarInputBatch {
 public:
  CsvBatchRef(const LineInputBatch& batch, char delimiter,
              std::optional<uint32_t> expected_num_columns)
      : _batch(batch),
        _ref_batch(_batch.size()),
        _delimiter(delimiter),
        _expected_num_columns(expected_num_columns) {}

  ColumnarInputSample& sample(uint32_t index) final {
    if (!_ref_batch.at(index)) {
      _ref_batch.at(index) =
          CsvSampleRef(_batch.at(index), _delimiter, _expected_num_columns);
    }
    return *_ref_batch.at(index);
  }

  uint32_t size() final { return _batch.size(); }

 private:
  const LineInputBatch& _batch;
  std::vector<std::optional<CsvSampleRef>> _ref_batch;
  char _delimiter;
  std::optional<uint32_t> _expected_num_columns;
};
}  // namespace thirdai::dataset