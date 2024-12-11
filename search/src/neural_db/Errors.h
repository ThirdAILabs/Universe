#pragma once

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



}  // namespace thirdai::search::ndb