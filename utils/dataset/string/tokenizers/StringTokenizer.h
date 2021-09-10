#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::utils {
  class StringTokenizer {
    public:
    virtual void tokenize(const std::string& str, std::unordered_map<uint32_t, float>& hashes, std::vector<uint32_t>& indices, std::vector<float>& values);
  };
}