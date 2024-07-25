#include "utils.h"
#include <algorithm>
#include <vector>

namespace thirdai::data::ner::utils {
bool isNumberWithPunct(const std::string& s,
                       const std::unordered_set<char>& exception_chars) {
  bool has_digit = false;

  for (char c : s) {
    if (std::isdigit(c)) {
      has_digit = true;
    } else if (!std::ispunct(c) && exception_chars.count(c) == 0) {
      return false;
    }
  }

  return has_digit;
}

bool containsAlphabets(const std::string& input,
                       const std::unordered_set<char>& exception_chars) {
  return std::any_of(input.begin(), input.end(), [exception_chars](char c) {
    return std::isalpha(c) && exception_chars.count(c) == 0;
  });
}

bool containsNumbers(const std::string& input,
                     const std::unordered_set<char>& exception_chars) {
  return std::any_of(input.begin(), input.end(), [exception_chars](char c) {
    return std::isdigit(c) && exception_chars.count(c) == 0;
  });
}

std::string trimPunctuation(const std::string& str) {
  const std::string punctuation = ".,?-!;:[]{}&%";
  size_t start = str.find_first_not_of(punctuation);
  if (start == std::string::npos) {
    return str;
  }
  size_t end = str.find_last_not_of(punctuation);
  return str.substr(start, end - start + 1);
}

std::vector<std::string> cleanAndLowerCase(
    const std::vector<std::string>& tokens) {
  /*
   * Converts the tokens to lower case and trims punctuations.
   */
  auto lower_tokens = tokens;
  for (auto& token : lower_tokens) {
    for (char& c : token) {
      c = std::tolower(c);
    }
  }
  for (auto& token : lower_tokens) {
    token = trimPunctuation(token);
  }
  return lower_tokens;
}

std::string stripNonDigits(const std::string& input) {
  std::string digits;
  for (char ch : input) {
    if (std::isdigit(ch)) {
      digits += ch;
    }
  }
  return digits;
}

bool containsKeywordInRange(const std::vector<std::string>& tokens,
                            size_t start, size_t end,
                            const std::unordered_set<std::string>& keywords) {
  return std::any_of(tokens.begin() + start, tokens.begin() + end,
                     [&keywords](const std::string& token) {
                       return keywords.find(token) != keywords.end();
                     });
}

bool luhnCheck(const std::string& number) {
  /*
   * Checks whether the number being passed satisifies the luhn's check. This is
   * useful for detecting credit card numbers.
   */
  int sum = 0;
  bool alternate = false;
  for (int i = number.size() - 1; i >= 0; --i) {
    int n = number[i] - '0';
    if (alternate) {
      n *= 2;
      if (n > 9) {
        n -= 9;
      }
    }
    sum += n;
    alternate = !alternate;
  }
  return (sum % 10 == 0);
}

std::string findContiguousNumbers(const std::vector<std::string>& v,
                                  uint32_t index, uint32_t k) {
  /*
   * Returns the surrounding numbers around the target token as a space
   * seperated string. This is useful when we have tokens of the form 1234 5678
   * 9101.
   */
  if (index >= static_cast<uint32_t>(v.size())) {
    return "";
  }

  if (!isNumberWithPunct(v[index], {'e', 'x', 't'})) {
    return "";
  }

  int start = index > k ? index - k : 0;
  int end = std::min(static_cast<uint32_t>(v.size()) - 1, index + k);

  std::vector<std::string> left_window, right_window;
  for (int i = index - 1; i >= start; --i) {
    if (isNumberWithPunct(v[index], {'e', 'x', 't'})) {
      left_window.push_back(v[i]);
    } else {
      break;
    }
  }

  std::reverse(left_window.begin(), left_window.end());

  for (int i = index + 1; i <= end; ++i) {
    if (isNumberWithPunct(v[index], {'e', 'x', 't'})) {
      right_window.push_back(v[i]);
    } else {
      break;
    }
  }
  if (left_window.empty() && right_window.empty()) {
    return "";
  }

  std::string result;
  for (const auto& s : left_window) {
    result += s + " ";
  }
  result += v[index] + " ";
  for (const auto& s : right_window) {
    result += s + " ";
  }

  return result;
}

uint32_t findMaxContiguousWindow(const SentenceTags& sentence_tags,
                                 uint32_t index,
                                 const std::string& tag_to_find) {
  int count = 0;

  // Check right from the index
  for (size_t i = index; i < sentence_tags.size(); ++i) {
    if (sentence_tags[i].empty() || sentence_tags[i][0].first != tag_to_find) {
      break;
    }
    count++;
  }

  return count;
}
}  // namespace thirdai::data::ner::utils