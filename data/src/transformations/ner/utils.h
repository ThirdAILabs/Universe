#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <unordered_set>

namespace thirdai::data {

inline std::string trimPunctuation(const std::string& str) {
  const std::string punctuation = ".,?-!;:";
  size_t start = str.find_first_not_of(punctuation);
  if (start == std::string::npos) {
    return str;
  }
  size_t end = str.find_last_not_of(punctuation);
  return str.substr(start, end - start + 1);
}

inline std::vector<std::string> toLowerCaseTokens(
    const std::vector<std::string>& tokens) {
  std::vector<std::string> lower_tokens(tokens.size());
  std::transform(tokens.begin(), tokens.end(), lower_tokens.begin(),
                 [](const std::string& token) {
                   std::string lower_token;
                   lower_token.reserve(token.size());
                   std::transform(token.begin(), token.end(),
                                  std::back_inserter(lower_token), ::tolower);
                   return lower_token;
                 });
  for (auto& token : lower_tokens) {
    token = trimPunctuation(token);
  }
  return lower_tokens;
}

inline bool containsKeywordInRange(
    const std::vector<std::string>& tokens, size_t start, size_t end,
    const std::unordered_set<std::string>& keywords) {
  return std::any_of(tokens.begin() + start, tokens.begin() + end,
                     [&keywords](const std::string& token) {
                       return keywords.find(token) != keywords.end();
                     });
}

inline std::string stripNonDigits(const std::string& input) {
  std::string digits;
  for (char ch : input) {
    if (std::isdigit(ch)) {
      digits += ch;
    }
  }
  return digits;
}

inline bool containsAlphabets(const std::string& input) {
  return std::any_of(input.begin(), input.end(), ::isalpha);
}

inline bool is_number(const std::string& s) {
  return std::all_of(s.begin(), s.end(), ::isdigit);
}

inline bool luhnCheck(const std::string& number) {
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

inline std::string find_contiguous_numbers(const std::vector<std::string>& v,
                                           uint32_t index, uint32_t k = 3) {
  if (index >= static_cast<uint32_t>(v.size())) {
    return "";
  }

  if (!is_number(v[index])) {
    return "";
  }

  int start = index > k ? index - k : 0;
  int end = std::min(static_cast<uint32_t>(v.size()) - 1, index + k);

  std::vector<std::string> left_window, right_window;
  for (int i = index - 1; i >= start; --i) {
    if (is_number(v[i])) {
      left_window.push_back(v[i]);
    } else {
      break;
    }
  }

  std::reverse(left_window.begin(), left_window.end());

  for (int i = index + 1; i <= end; ++i) {
    if (is_number(v[i])) {
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
    result += s;
  }
  result += v[index];
  for (const auto& s : right_window) {
    result += s;
  }

  // Remove trailing space
  if (!result.empty()) {
    result.pop_back();
  }

  return result;
}

inline std::string getNumericalFeatures(const std::string& input) {
  std::string strippedInput = stripNonDigits(input);

  if (!strippedInput.empty()) {
    if (luhnCheck(strippedInput) || strippedInput.size() > 12) {
      return "IS_ACCOUNT_NUMBER";
    }

    if (containsAlphabets(input) && input.size() >= 6) {
      return "IS_UIN";
    }

    if ((strippedInput.size() >= 9 && strippedInput.size() <= 12) ||
        input[0] == '+' || input[0] == '(' || input.back() == ')') {  // NOLINT
      return "MAYBE_PHONE";
    }

    if (strippedInput.size() == input.size() && strippedInput.size() >= 5) {
      return "IS_NUMBER_OR_UIN";
    }

    if ((strippedInput.size() <= 2 && std::stoi(strippedInput) <= 31) ||
        strippedInput.size() == 4) {
      return "A_DATE";
    }

    if (strippedInput.size() <= 6 && strippedInput.size() >= 5) {
      return "MAYBE_ZIP_CODE";
    }
  }
  return "";
}

}  // namespace thirdai::data