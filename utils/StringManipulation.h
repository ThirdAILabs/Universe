#pragma once
#include <string>
#include <string_view>
#include <vector>

// TODO(any): consolidate string manipulation to this file
namespace thirdai::text {

/**
 * Splits a string by delimiter.
 */
std::vector<std::string_view> split(std::string_view string, char delimiter);

/**
 * Parses a sentence into tokens.
 */
std::vector<std::string_view> tokenizeSentence(std::string_view string);

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

}  // namespace thirdai::text
