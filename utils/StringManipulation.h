#pragma once
#include <string>
#include <vector>

namespace thirdai::text {

/**
 * Splits a string by delimiter.
 */
std::vector<std::string> split(const std::string& string, char delimiter);

/**
 * Parses a sentence into word and punctuation tokens.
 * The returned tokens are strings, not to be confused with integer token IDs,
 * For example, "Anshu is CEO, Tharun is CTO." will be parsed into:
 * {"Anshu", "is", "CEO", ",", "Tharun", "is", "CTO", "."}
 *
 * Note: this function does not currently support no-latin alphabet characters.
 * To see more expected behaviors, see tests prefixed with
 * "TestTokenizeSentence" in StringManipulationTests.cc
 */
std::vector<std::string> tokenizeSentence(const std::string& sentence);

std::vector<std::string> charKGrams(const std::string& text, uint32_t k);

/**
 * Joins a vector of strings into a single delimited string.
 */
std::string join(const std::vector<std::string>& strings,
                 const std::string& delimiter);

/**
 * Creates a copy of the original stringview where all characters are lowercase.
 */
inline std::string lower(const std::string& str) {
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
