#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::utils {
class StringTokenizer {
 public:
  virtual void getTokenIds(const std::string& str,
                        std::vector<uint32_t>& indices,
                        std::vector<float>& values){};
};
}  // namespace thirdai::utils