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

bool isControl(const wchar_t& c);
bool isWhitespace(const wchar_t& c);
bool isPunctuation(const wchar_t& c);
bool isChineseChar(const wchar_t& c);

const std::wstring DEFAULT_STRIP_CHARACTERS = L" \t\n\r\v\f";

std::wstring join(const std::vector<std::wstring>& atoms,
                  const std::wstring& delimiter);

template <class Predicate>
std::vector<std::wstring> splitIf(const std::wstring& text,
                                  Predicate predicate);
std::vector<std::wstring> split(
    const std::wstring& text,
    const std::wstring& split_characters = DEFAULT_STRIP_CHARACTERS);

std::wstring strip(
    const std::wstring& text,
    const std::wstring& strip_characters = DEFAULT_STRIP_CHARACTERS);

std::string convertFromUnicode(const std::wstring& wText);
std::wstring convertToUnicode(const std::string& text);
std::string normalizeNFD(const std::string& s);
std::wstring lower(const std::wstring& s);

std::wstring normalizeSpaces(const std::wstring& text);
std::wstring stripAccents(const std::wstring& text);

std::vector<std::wstring> splitOnPunctuation(const std::wstring& text);
std::vector<std::wstring> splitOnWhitespace(const std::wstring& text);

}  // namespace thirdai::text
