#pragma once

#include <_types/_uint32_t.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

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
 * representations of a columnar input sample.
 * For example, a columnar input sample can be a vector of strings, where each
 * element is a column, or a map from column names to their values.
 */
class ColumnarInputSample {
 public:
  // TODO(Geordie): Add methods to parse column into their real types?
  // E.g. "1" -> int(1)

  /**
   * Returns the identified column (either by column number or column name)
   * as a string view.
   */
  virtual std::string_view column(const ColumnIdentifier& column) = 0;
  /**
   * The size of the columnar sample; the number of columns.
   */
  virtual uint32_t size() = 0;
  virtual ~ColumnarInputSample() = default;
};

/**
 * A wrapper around a reference to a columnar sample represented by a map from
 * column names (string) to column values (string). Implements the
 * ColumnarInputSample interface.
 *
 * It wraps around a reference instead of the object directly to prevent copying
 * overheads.
 */
class MapSampleRef final : public ColumnarInputSample {
 public:
  explicit MapSampleRef(const MapInput& columns) : _columns(columns) {}

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

/**
 * A wrapper around a reference to a columnar sample represented by a vector
 * of string views. Implements the ColumnarInputSample interface.
 *
 * It wraps around a reference instead of the object directly to prevent copying
 * overheads.
 */
class RowSampleRef final : public ColumnarInputSample {
 public:
  explicit RowSampleRef(const RowInput& columns) : _columns(columns) {}

  std::string_view column(const ColumnIdentifier& column) final {
    if (column.number() >= _columns.size()) {
      std::stringstream error;
      error << "Tried to access " << column.number()
            << "-th column but this row only has " << _columns.size()
            << " columns:" << std::endl;
      for (auto column : _columns) {
        error << " \"" << column << "\"";
      }
      throw std::invalid_argument(error.str());
    }
    return _columns.at(column.number());
  }

  uint32_t size() final { return _columns.size(); }

 private:
  const RowInput& _columns;
};

/**
 * A wrapper around a reference to a columnar sample represented by a CSV
 * string. Implements the ColumnarInputSample interface.
 *
 * It wraps around a reference instead of the object directly to prevent copying
 * overheads.
 */
class CsvSampleRef final : public ColumnarInputSample {
 public:
  CsvSampleRef(const LineInput& line, char delimiter,
               std::optional<uint32_t> expected_num_cols = std::nullopt)
      : _columns(ProcessorUtils::parseCsvRow(line, delimiter)) {
    if (expected_num_cols && expected_num_cols != _columns.size()) {
      std::stringstream error;
      error << "Expected " << *expected_num_cols
            << " columns in each row of the dataset. Found row with "
            << _columns.size() << " columns:";
      for (auto column : _columns) {
        error << " \"" << column << "\"";
      }
      throw std::invalid_argument(error.str());
    }
  }

  explicit CsvSampleRef(std::vector<std::string_view> line)
      : _columns(std::move(line)) {}

  std::string_view at(uint32_t index) { return _columns[index]; }

  std::string_view column(const ColumnIdentifier& column) final {
    return RowSampleRef(_columns).column(column);
  }

  void insert(const RowInput& values) {
    _columns.insert(_columns.end(), values.begin(), values.end());
  }

  uint32_t size() final { return _columns.size(); }

 private:
  RowInput _columns;
};

/**
 * An interface that allows the data pipeline to operate on arbitrary
 * representations of a columnar input batch.
 * For example, a columnar input batch can be a vector of maps from column names
 * to their values, or a vector of CSV strings.
 * ColumnarInputBatch is essentially a vector of ColumnarInputSamples, but we
 * chose not to use a vector like that for two reasons:
 * 1. There cannot be a vector of abstract classes; we will have to use
 * pointers.
 * 2. This gives us the flexibility to accomodate more representations and
 * include custom logic as needed.
 */
class ColumnarInputBatch {
 public:
  /**
   * Gets a reference to the index-th sample in the batch.
   */
  virtual ColumnarInputSample& at(uint32_t index) = 0;
  /**
   * Returns the size of the batch.
   */
  virtual uint32_t size() const = 0;
  virtual ~ColumnarInputBatch() = default;

 protected:
  void assertIndexInRange(uint32_t index) const {
    if (size() <= index) {
      std::stringstream error;
      error << "Attempted to access " << index
            << "-th sample but this batch only contains " << size()
            << " samples.";
      throw std::invalid_argument(error.str());
    }
  }
};

/**
 * A wrapper around a reference to a vector of columnar samples represented by
 * maps from column names (string) to column values (string). Implements the
 * ColumnarInputSample interface.
 *
 * It wraps around a reference instead of the object directly to prevent copying
 * overheads.
 */
class MapBatchRef final : public ColumnarInputBatch {
 public:
  // NOLINTNEXTLINE
  MapBatchRef(const MapInputBatch& batch) : _batch(batch) {
    // We only store references so this is very fast.
    _ref_batch.reserve(_batch.size());
    for (const auto& input : _batch) {
      _ref_batch.emplace_back(MapSampleRef(input));
    }
  }

  ColumnarInputSample& at(uint32_t index) final {
    assertIndexInRange(index);
    return _ref_batch.at(index);
  }

  uint32_t size() const final { return _batch.size(); }

 private:
  const MapInputBatch& _batch;
  std::vector<MapSampleRef> _ref_batch;
};

/**
 * A wrapper around a reference to a vector of columnar samples represented by
 * CSV strings. Implements the ColumnarInputSample interface.
 *
 * It wraps around a reference instead of the object directly to prevent copying
 * overheads.
 */
class CsvBatchRef final : public ColumnarInputBatch {
 public:
  CsvBatchRef(const LineInputBatch& batch, char delimiter,
              std::optional<uint32_t> expected_num_columns)
      : _batch(batch),
        _ref_batch(_batch.size()),
        _delimiter(delimiter),
        _expected_num_columns(expected_num_columns) {}

  ColumnarInputSample& at(uint32_t index) final {
    checkCorrectness(index);
    return *_ref_batch.at(index);
  }

  CsvSampleRef& get(uint32_t index) {
    checkCorrectness(index);
    return *_ref_batch.at(index);
  }

  void insert(uint32_t index, const RowInput& values) {
    checkCorrectness(index);
    _ref_batch.at(index)->insert(values);
  }

  void checkCorrectness(uint32_t index) {
    assertIndexInRange(index);
    /*
      Constructing CsvSampleRef also parses the CSV string. This is an
      expensive operation, so we delay it until the caller demands the sample
      (lazy execution). Additionally, if called in a parallel region, then
      the parsing process is also parallelized across samples.
    */
    if (!_ref_batch.at(index)) {
      _ref_batch.at(index) =
          CsvSampleRef(_batch.at(index), _delimiter, _expected_num_columns);
    }
  }

  uint32_t size() const final { return _batch.size(); }

 private:
  const LineInputBatch& _batch;
  std::vector<std::optional<CsvSampleRef>> _ref_batch;
  char _delimiter;
  std::optional<uint32_t> _expected_num_columns;
};

class CsvRolledBatch final : public ColumnarInputBatch {
 public:
  explicit CsvRolledBatch(
      const std::vector<std::vector<std::string_view>>& rows)
      : _batch_values(rows.size()) {
    for (uint32_t i = 0; i < rows.size(); i++) {
      _batch_values[i] = CsvSampleRef(rows[i]);
    }
  }

  ColumnarInputSample& at(uint32_t index) final { return _batch_values[index]; }

  uint32_t size() const final { return _batch_values.size(); }

 private:
  std::vector<CsvSampleRef> _batch_values;
};
}  // namespace thirdai::dataset