#pragma once

#include <utils/Random.h>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace thirdai::text {

/**
 * Splits a string by delimiter.
 */
std::vector<std::string> split(const std::string_view& string, char delimiter);

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
std::vector<std::string> tokenizeSentence(const std::string_view& sentence);

std::vector<std::string> charKGrams(const std::string_view& text, uint32_t k);

std::vector<std::string> wordLevelCharKGrams(
    const std::vector<std::string>& words, uint32_t k,
    size_t min_word_length = 4);

std::string deleteRandomCharacters(const std::string& input,
                                   size_t num_to_delete, std::mt19937& rng);

std::string duplicateRandomCharacters(const std::string& input,
                                      size_t num_to_duplicate,
                                      std::mt19937& rng);

std::string replaceRandomCharactersWithSpaces(const std::string& input,
                                              size_t num_to_replace,
                                              std::mt19937& rng);

std::string replaceRandomCharactersWithKeyboardAdjacents(
    const std::string& input, size_t num_to_replace, std::mt19937& rng);

std::string perturbCharacters(const std::string& input,
                              size_t chars_replace_with_space,
                              size_t chars_deleted, size_t chars_duplicated,
                              size_t chars_replace_with_adjacents,
                              std::mt19937& rng);

/**
 * Joins a vector of strings into a single delimited string.
 */
std::string join(const std::vector<std::string>& strings,
                 const std::string& delimiter);

/**
 * Creates a copy of the original stringview where all characters are lowercase.
 */
inline std::string lower(const std::string_view& str) {
  std::string lower_name;
  for (const char c : str) {
    lower_name.push_back(std::tolower(c));
  }
  return lower_name;
}

const std::string DEFAULT_STRING_STRIP_CHARACTERS = " \t\n\r\v\f";

/**
 *Strips leading and tailing whitespace.
 */
std::string stripWhitespace(
    const std::string& s,
    const std::string& strip_characters = DEFAULT_STRING_STRIP_CHARACTERS);

/**
 * Extracts an integer value from an integer string.
 */
inline uint32_t toInteger(const char* start) {
  char* end;
  return std::strtoul(start, &end, 10);
}

bool startsWith(const std::string& to_search_in, const std::string& prefix);

/**
 * Replaces punctuation characters in string with a given character.
 */
std::string replacePunctuation(std::string string, char replace_char = ' ');

/**
 * Replaces \n and \r characters in string with a given character.
 */
std::string replaceNewlines(std::string string, char replace_char = ' ');

/* HELPER METHODS FOR UNICODE STRINGS */

std::wstring toUnicode(const std::string& text);
std::string fromUnicode(const std::wstring& wText);

/**
 * Converts the given input string into a consistent normalized form in case of
 * unicode text. Read more about normalization here:
 * https://unicode.org/faq/normalization.html
 */
std::string normalize(const std::string& s);

std::wstring lower(const std::wstring& s);

const std::wstring DEFAULT_UNICODE_STRIP_CHARACTERS = L" \t\n\r\v\f";

std::wstring strip(
    const std::wstring& text,
    const std::wstring& strip_characters = DEFAULT_UNICODE_STRIP_CHARACTERS);

std::vector<std::wstring> split(
    const std::wstring& text,
    const std::wstring& split_characters = DEFAULT_UNICODE_STRIP_CHARACTERS);

template <class Predicate>
std::vector<std::wstring> splitIf(const std::wstring& text,
                                  Predicate predicate);

/**
 * Cleans the text by doing things like stripping accents, replacing common
 * whitespace characters with a space, and cleaning out extraneous control
 * charcters, null characters, and unrepresentable characters.
 */
std::wstring cleanText(const std::wstring& text);

bool isControl(const wchar_t& c);
bool isWhitespace(const wchar_t& c);
bool isPunctuation(const wchar_t& c);
bool isChineseChar(const wchar_t& c);

std::vector<std::wstring> tokenizeByPunctuations(const std::wstring& text);
std::vector<std::wstring> splitOnWhitespace(const std::wstring& text);

}  // namespace thirdai::text
