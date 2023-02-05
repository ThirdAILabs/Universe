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

const std::wstring DEFAULT_STRIP_CHARACTERS = L" \t\n\r\v\f";

class is_any_of {
 public:
  explicit is_any_of(std::wstring delimiters);
  bool operator()(wchar_t candidate) const;

 private:
  std::wstring delimiters_;
};

std::wstring join(const std::vector<std::wstring>& atoms,
                  const std::wstring& delimiter);

template <class Predicate>
void split(std::vector<std::wstring>& result, const std::wstring& s,
           Predicate predicate);
//
std::string convertFromUnicode(const std::wstring& wText);
std::wstring convertToUnicode(const std::string& text);
std::string normalize_nfd(const std::string& s);
std::wstring tolower(const std::wstring& s);
std::vector<std::wstring> split(const std::wstring& text);
std::vector<std::wstring> whitespaceTokenize(const std::wstring& text);
std::wstring strip(const std::wstring& text);

bool isControl(const wchar_t& ch);
bool isWhitespace(const wchar_t& ch);
bool isPunctuation(const wchar_t& ch);
bool isChineseChar(const wchar_t& ch);
bool isStripChar(const wchar_t& ch);

}  // namespace thirdai::text
