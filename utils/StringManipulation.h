#include <string>

namespace thirdai::utils {

/**
 * Creates a copy of the original string where all characters are lowercase.
 */
inline std::string lower(const std::string& str) {
  std::string lower_name;
  for (char c : str) {
    lower_name.push_back(std::tolower(c));
  }
  return lower_name;
}
}  // namespace thirdai::utils
