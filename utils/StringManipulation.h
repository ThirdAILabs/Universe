#pragma once
#include <string>
#include <string_view>
#include <vector>

// TODO(any): consolidate string manipulation to this file
namespace thirdai::text {

/**
 * Splits a sentence by delimiter.
 */
std::vector<std::string_view> split(std::string_view sentence,
                                    char delimiter = ' ');

/**
 * Joins a vector of strings into a single delimited string.
 */
std::string join(const std::vector<std::string>& strings,
                 const std::string& delimiter);

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

bool startsWith(const std::string& to_search_in, const std::string& prefix);

}  // namespace thirdai::text
