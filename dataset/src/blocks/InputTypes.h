#pragma once

#include <dataset/src/blocks/ColumnIdentifier.h>
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

class SingleInputRef {
 public:
  virtual std::string_view column(const ColumnIdentifier& column) = 0;
  virtual uint32_t size() = 0;
  virtual std::exception_ptr assertValid(uint32_t expected_num_cols) = 0;
  virtual ~SingleInputRef() = default;
};

class SingleMapInputRef final : public SingleInputRef {
 public:
  // NOLINTNEXTLINE Implicit constructor is intentional
  SingleMapInputRef(const MapInput& columns) : _columns(columns) {}

  std::string_view column(const ColumnIdentifier& column) final {
    if (!_columns.count(column.name())) {
      return {};
    }
    return _columns.at(column.name());
  }

  uint32_t size() final { return _columns.size(); }

  std::exception_ptr assertValid(uint32_t expected_num_cols) final {
    (void)expected_num_cols;
    return nullptr;
  }

 private:
  const MapInput& _columns;
};

class SingleRowInputRef final : public SingleInputRef {
 public:
  // NOLINTNEXTLINE Implicit constructor is intentional
  SingleRowInputRef(const RowInput& columns) : _columns(columns) {}

  std::string_view column(const ColumnIdentifier& column) final {
    return _columns.at(column.number());
  }

  std::exception_ptr assertValid(uint32_t expected_num_cols) final {
    if (_columns.size() == expected_num_cols) {
      return nullptr;
    }

    std::stringstream error_ss;
    error_ss << "Expected " << expected_num_cols
             << " columns in each row of the dataset. Found row with "
             << _columns.size() << " columns:";
    for (auto column : _columns) {
      error_ss << " '" << column << "'";
    }
    error_ss << ".";

    return std::make_exception_ptr(std::invalid_argument(error_ss.str()));
  }

  uint32_t size() final { return _columns.size(); }

 private:
  const RowInput& _columns;
};

class SingleCsvLineInputRef final : public SingleInputRef {
 public:
  SingleCsvLineInputRef(const LineInput& line, char delimiter)
      : _line(line), _delimiter(delimiter) {}

  std::string_view column(const ColumnIdentifier& column) final {
    parseToColumnsIfNecessary();
    return SingleRowInputRef(*_columns).column(column);
  }

  uint32_t size() final {
    parseToColumnsIfNecessary();
    return _columns->size();
  }

  std::exception_ptr assertValid(uint32_t expected_num_cols) final {
    parseToColumnsIfNecessary();
    return SingleRowInputRef(*_columns).assertValid(expected_num_cols);
  }

 private:
  void parseToColumnsIfNecessary() {
    if (!_columns) {
      _columns = ProcessorUtils::parseCsvRow(/* row= */ _line,
                                             /* delimiter= */ _delimiter);
    }
  }

  const LineInput& _line;
  char _delimiter;
  std::optional<RowInput> _columns;
  std::exception_ptr _last_error;
};

class BatchInputRef {
 public:
  virtual SingleInputRef& sample(uint32_t index) = 0;
  virtual uint32_t size() = 0;
  virtual ~BatchInputRef() = default;
};

class BatchMapInputRef final : public BatchInputRef {
 public:
  // NOLINTNEXTLINE
  BatchMapInputRef(const MapInputBatch& batch) : _batch(batch) {
    _ref_batch.reserve(_batch.size());
    for (const auto& input : _batch) {
      _ref_batch.emplace_back(SingleMapInputRef(input));
    }
  }

  SingleInputRef& sample(uint32_t index) final { return _ref_batch.at(index); }

  uint32_t size() final { return _batch.size(); }

 private:
  const MapInputBatch& _batch;
  std::vector<SingleMapInputRef> _ref_batch;
};

class BatchCsvLineInputRef final : public BatchInputRef {
 public:
  BatchCsvLineInputRef(const LineInputBatch& batch, char delimiter)
      : _batch(batch) {
    _ref_batch.reserve(_batch.size());
    for (const auto& input : _batch) {
      _ref_batch.emplace_back(SingleCsvLineInputRef(input, delimiter));
    }
  }

  SingleInputRef& sample(uint32_t index) final { return _ref_batch.at(index); }

  uint32_t size() final { return _batch.size(); }

 private:
  const LineInputBatch& _batch;
  std::vector<SingleCsvLineInputRef> _ref_batch;
};
}  // namespace thirdai::dataset