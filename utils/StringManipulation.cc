#include "StringManipulation.h"
#include <cctype>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace thirdai::text {

std::vector<std::string_view> split(std::string_view string, char delimiter) {
  std::vector<std::string_view> words;

  bool prev_is_delim = true;
  uint32_t start_of_word_offset = 0;
  for (uint32_t i = 0; i < string.size(); i++) {
    if (prev_is_delim && string[i] != delimiter) {
      // If we go from a space to a non-space character then we are at the
      // start of a word.
      start_of_word_offset = i;
      prev_is_delim = false;
    }
    if (!prev_is_delim && string[i] == delimiter) {
      // If we go from a non-space character to a space then we are at the end
      // of a word.
      uint32_t len = i - start_of_word_offset;

      std::string_view word_view(string.data() + start_of_word_offset, len);

      words.push_back(word_view);
      prev_is_delim = true;
    }
  }
  if (!prev_is_delim) {
    // If we don't find a space at the end of the sentence, then there's a
    // last word we need to hash.
    uint32_t len = string.size() - start_of_word_offset;

    std::string_view word_view(string.data() + start_of_word_offset, len);

    words.push_back(word_view);
  }

  return words;
}

namespace {

class LocalView {
 public:
  LocalView(std::string_view string, uint32_t pos)
      : _current_is_space(std::isspace(string[pos])),
        _current_is_punct(std::ispunct(string[pos])),
        _prev_not_space(false),
        _prev_not_punct(false),
        _next_not_space(false),
        _next_not_punct(false) {
    if (pos > 0) {
      _prev_not_space = !std::isspace(string[pos - 1]);
      _prev_not_punct = !std::ispunct(string[pos - 1]);
    }
    if (pos + 1 < string.size()) {
      _next_not_space = !std::isspace(string[pos + 1]);
      _next_not_punct = !std::ispunct(string[pos + 1]);
    }
  }

  bool currentIsSpace() const { return _current_is_space; }
  bool currentIsPunct() const { return _current_is_punct; }
  bool prevNotSpace() const { return _prev_not_space; }
  bool prevNotPunct() const { return _prev_not_punct; }
  bool nextNotSpace() const { return _next_not_space; }
  bool nextNotPunct() const { return _next_not_punct; }

 private:
  bool _current_is_space;
  bool _current_is_punct;
  bool _prev_not_space;
  bool _prev_not_punct;
  bool _next_not_space;
  bool _next_not_punct;
};

uint32_t startPosition(std::string_view string) {
  uint32_t position = 0;
  for (char c : string) {
    if (!std::isspace(c)) {
      break;
    }
    position++;
  }
  return position;
}

uint32_t endPosition(std::string_view string) {
  uint32_t position = string.size();
  for (auto it = string.rbegin(); it != string.rend(); it++) {
    if (!std::isspace(*it)) {
      break;
    }
    position--;
  }
  return position;
}

uint32_t countTokens(std::string_view trimmed_string) {
  uint32_t n_tokens = 1;
  for (uint32_t pos = 0; pos < trimmed_string.size(); pos++) {
    LocalView loc(trimmed_string, pos);
    if (loc.currentIsSpace() && loc.prevNotSpace()) {
      n_tokens++;
    }
    if (loc.currentIsPunct()) {
      if (loc.prevNotSpace()) {
        n_tokens++;
      }
      if (loc.nextNotPunct() && loc.nextNotSpace()) {
        n_tokens++;
      }
    }
  }
  return n_tokens;
}

class Tokens {
 public:
  explicit Tokens(uint32_t size) : _tokens(size), _index(0) {}

  void addToken(std::string_view token) {
    if (_index >= _tokens.size()) {
      throw std::runtime_error("More tokens in sentence than expected!");
    }
    _tokens[_index] = token;
    _index++;
  }

  std::vector<std::string_view> tokens() { return std::move(_tokens); }

 private:
  std::vector<std::string_view> _tokens;
  uint32_t _index;
};

}  // namespace

std::vector<std::string_view> tokenizeSentence(std::string_view string) {
  uint32_t start = startPosition(string);
  if (string.empty() || start == string.size()) {
    return {};
  }

  string = string.substr(start, endPosition(string) - start);

  Tokens tokens(countTokens(string));
  uint32_t token_start = 0;

  for (uint32_t position = 0; position < string.size(); position++) {
    LocalView loc(string, position);
    if (loc.currentIsSpace()) {
      if (loc.prevNotSpace() && loc.prevNotPunct()) {
        tokens.addToken(string.substr(token_start, position - token_start));
      }
      token_start = position + 1;
    }
    if (loc.currentIsPunct()) {
      if (loc.prevNotSpace() && loc.prevNotPunct()) {
        tokens.addToken(string.substr(token_start, position - token_start));
      }
      tokens.addToken(string.substr(position, 1));
      token_start = position + 1;
    }
  }

  if (!std::ispunct(string.back())) {
    tokens.addToken(string.substr(token_start));
  }

  return tokens.tokens();
}

}  // namespace thirdai::text