#pragma once

#include <rocksdb/status.h>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::search::ndb {

enum class ErrorCode { DbError, DocNotFound, MalformedData, ReadOnly };

class NeuralDbError : public std::exception {
 public:
  NeuralDbError(ErrorCode code, std::string msg)
      : _code(code), _msg(std::move(msg)) {}

  ErrorCode code() const { return _code; }

  const char* what() const noexcept override { return _msg.c_str(); }

 private:
  ErrorCode _code;
  std::string _msg;
};

class RocksdbError : public NeuralDbError {
 public:
  RocksdbError(const rocksdb::Status& status, const std::string& action)
      : NeuralDbError(ErrorCode::DbError, format(status, action)) {}

 private:
  static std::string format(const rocksdb::Status& status,
                            const std::string& action) {
    auto msg = status.ToString();
    if (msg.back() == ' ') {
      msg.pop_back();
    }
    if (msg.back() == ':') {
      msg.pop_back();
    }
    return "db returned '" + msg + "' while " + action;
  }
};

}  // namespace thirdai::search::ndb