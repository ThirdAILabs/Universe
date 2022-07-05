#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

enum class SchemaKey {
  user,
  item,
  timestamp,
  text_attr,
  categorical_attr,
  trackable_quantity,
  target
};

using GivenSchema = std::unordered_map<std::string, std::string>;
using InternalSchema = std::unordered_map<SchemaKey, std::string>;
using ColumnNumbers = std::unordered_map<SchemaKey, size_t>;

class SchemaProcessor {
 public:
  explicit SchemaProcessor(GivenSchema& schema);

  ColumnNumbers parseHeader(const std::string& header, char delimiter);

 private:
  InternalSchema _schema;
};

}  // namespace thirdai::bolt