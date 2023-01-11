#pragma once

#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/utils/CsvParser.h>
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
  CsvSampleRef(const LineInput& line, const std::string& delimiter,
               std::optional<uint32_t> expected_num_cols = std::nullopt)
      : _columns(CSV::parse(line, delimiter)) {
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
  virtual ColumnarInputSample& sample(uint32_t index) = 0;
  /**
   * Returns the size of the batch.
   */
  virtual uint32_t size() = 0;
  virtual ~ColumnarInputBatch() = default;
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

  ColumnarInputSample& sample(uint32_t index) final {
    return _ref_batch.at(index);
  }

  uint32_t size() final { return _batch.size(); }

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
  CsvBatchRef(const LineInputBatch& batch, const std::string& delimiter,
              std::optional<uint32_t> expected_num_columns)
      : _batch(batch),
        _ref_batch(_batch.size()),
        _delimiter(delimiter),
        _expected_num_columns(expected_num_columns) {}

  ColumnarInputSample& sample(uint32_t index) final {
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
    return *_ref_batch.at(index);
  }

  uint32_t size() final { return _batch.size(); }

 private:
  const LineInputBatch& _batch;
  std::vector<std::optional<CsvSampleRef>> _ref_batch;
  const std::string& _delimiter;
  std::optional<uint32_t> _expected_num_columns;
};
}  // namespace thirdai::dataset