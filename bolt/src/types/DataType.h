

#include <_types/_uint64_t.h>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

namespace thirdai::bolt {

class SequentialClassifierDataType {
 public:
  virtual std::string name() = 0;
  virtual ~SequentialClassifierDataType() = default;
};

using SequentialClassifierDataTypePtr =
    std::shared_ptr<SequentialClassifierDataType>;

class CategoricalDataType final : public SequentialClassifierDataType {
 public:
  explicit CategoricalDataType(uint64_t n_unique) : _unique_items(n_unique) {}

  CategoricalDataType(uint64_t n_unique, std::string delim)
      : _unique_items(n_unique), _delimeter(std::move(delim)) {}

  static constexpr const char* DATA_TYPE_NAME = "categorical-datatype";
  std::string name() final { return DATA_TYPE_NAME; }

  constexpr uint64_t getUniqueItemsCount() const { return _unique_items; }

 private:
  uint64_t _unique_items;
  std::optional<std::string> _delimeter;
};

using CategoricalDataTypePtr = std::shared_ptr<CategoricalDataType>;

class NumericalDataType final : public SequentialClassifierDataType {
 public:
  NumericalDataType() {}

  static constexpr const char* DATA_TYPE_NAME = "numerical-datatype";
  std::string name() final { return DATA_TYPE_NAME; }
};

using NumericalDataTypePtr = std::shared_ptr<NumericalDataType>;

class TextualDataType final : public SequentialClassifierDataType {
 public:
  TextualDataType() {}

  static constexpr const char* DATA_TYPE_NAME = "text-datatype";
  std::string name() final { return DATA_TYPE_NAME; }
};

using TextualDataTypePtr = std::shared_ptr<TextualDataType>;

class DateTime final : public SequentialClassifierDataType {
 public:
  explicit DateTime(std::string date) : _date(std::move(date)) {}

  static constexpr const char* DATA_TYPE_NAME = "datetime-datatype";
  std::string name() final { return DATA_TYPE_NAME; }

  std::string date() const { return _date; }

 private:
  std::string _date;
};

using DateTimePtr = std::shared_ptr<DateTime>;

}  // namespace thirdai::bolt