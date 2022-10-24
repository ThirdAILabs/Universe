#pragma once
#include <string>
#include <string_view>

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

/**
 * Creates a copy of the original stringview where all characters are lowercase.
 */
inline std::string lower(const std::string_view str) {
  std::string lower_name;
  for (char c : str) {
    lower_name.push_back(std::tolower(c));
  }
  return lower_name;
}

/**
 * Extracts an integer value from an integer string.
 */
inline uint32_t toInteger(const char* start) {
  char* end;
  return std::strtoul(start, &end, 10);
}

}  // namespace thirdai::utils
