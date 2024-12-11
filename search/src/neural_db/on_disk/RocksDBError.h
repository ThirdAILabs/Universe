#pragma once

#include <rocksdb/status.h>
#include <search/src/neural_db/Errors.h>

namespace thirdai::search::ndb {

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