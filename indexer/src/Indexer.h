#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace thirdai::deployments {

class Indexer {
 public:
  explicit Indexer(const std::string& config_file_path)
      : _config_file_path(config_file_path) {}

 private:
  void constructFlash() {
    if (!_config_file_path) {
      throw std::invalid_argument("");
    }
  }

  std::optional<std::string> _config_file_path;
};

}  // namespace thirdai::deployments