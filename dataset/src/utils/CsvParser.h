#include <cstdint>
#include <exception>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace thirdai::dataset {

enum class DelimiterMatchState { Mismatch, HalfMatch, Matched };

struct DelimiterMatcher {
 public:
  explicit DelimiterMatcher(const std::string& delimiter)
      : _delimiter(delimiter), _pos(0) {}

  DelimiterMatchState matchDelimiter(char character) {
    if (character != _delimiter[_pos]) {
      _pos = 0;
      return DelimiterMatchState::Mismatch;
    }
    if (_pos == _delimiter.size() - 1) {
      _pos = 0;
      return DelimiterMatchState::Matched;
    }
    _pos++;
    return DelimiterMatchState::HalfMatch;
  }

 private:
  const std::string& _delimiter;
  uint32_t _pos;
};

enum class ParsingState {
  Start,
  InUnquotedColumn,
  InQuotedColumn,
  FoundQuoteInQuotedColumn,
  MatchedPartOfDelimiter,
  MatchedDelimiter
};

class ColumnParser {
 public:
  ColumnParser(std::string::const_iterator begin,
               const std::string::const_iterator& end,
               const std::string& delimiter)
      : _column_begin_candidate(begin),
        _column_end_candidate(begin),
        _next_column_start_parse(begin) {
    ParsingState state = ParsingState::Start;
    DelimiterMatcher delimiter_matcher(delimiter);

    for (auto char_iter = begin; char_iter < end; char_iter++) {
      state = nextState(state, *char_iter, delimiter_matcher);

      switch (state) {
        case ParsingState::InUnquotedColumn:
          _column_begin_candidate = begin;
          _column_end_candidate = char_iter + 1;
          break;
        case ParsingState::InQuotedColumn:
          _column_begin_candidate = begin + 1;
          _column_end_candidate = char_iter + 1;
          break;
        case ParsingState::MatchedDelimiter:
          _next_column_start_parse = char_iter + 1;
          return;
        default:
          break;
      }
    }

    // Only reach here if we don't match the delimiter after reaching
    // the end of the string.
    _next_column_start_parse = end;
  }

  auto begin() { return _column_begin_candidate; }
  auto end() { return _column_end_candidate; }
  auto nextColumnStartParse() { return _next_column_start_parse; }

 private:
  static ParsingState nextState(ParsingState previous_state, char character,
                                DelimiterMatcher& delimiter_matcher) {
    switch (previous_state) {
      case ParsingState::Start:
      case ParsingState::FoundQuoteInQuotedColumn:
        if (character == '"') {
          return ParsingState::InQuotedColumn;
        }
        return nextStateFromOutsideQuotes(character, delimiter_matcher);

      case ParsingState::InUnquotedColumn:
      case ParsingState::MatchedPartOfDelimiter:
        return nextStateFromOutsideQuotes(character, delimiter_matcher);

      case ParsingState::InQuotedColumn:
        if (character == '"') {
          return ParsingState::FoundQuoteInQuotedColumn;
        }
        return ParsingState::InQuotedColumn;

      case ParsingState::MatchedDelimiter:
        throw std::invalid_argument(
            "Cannot transition from MatchedWholeDelimiter.");
    }
    throw std::invalid_argument("Invalid state.");
  }

  static ParsingState nextStateFromOutsideQuotes(
      char character, DelimiterMatcher& delimiter_matcher) {
    switch (delimiter_matcher.matchDelimiter(character)) {
      case DelimiterMatchState::Mismatch:
        return ParsingState::InUnquotedColumn;
      case DelimiterMatchState::HalfMatch:
        return ParsingState::MatchedPartOfDelimiter;
      case DelimiterMatchState::Matched:
        return ParsingState::MatchedDelimiter;
    }
  }

  std::string::const_iterator _column_begin_candidate;
  std::string::const_iterator _column_end_candidate;
  std::string::const_iterator _next_column_start_parse;
};

class CsvRowParser {
 public:
  explicit CsvRowParser(const std::string& delimiter) : _delimiter(delimiter) {
    verifyDelimiterCorrectness(_delimiter);
  }

  std::vector<std::string_view> parse(const std::string& line) const {
    std::vector<std::string_view> row;

    auto column_start_parse = line.begin();
    while (column_start_parse != line.end()) {
      ColumnParser column(column_start_parse, line.end(), _delimiter);
      row.emplace_back(column.begin().base(),
                       std::distance(column.begin(), column.end()));
      column_start_parse = column.nextColumnStartParse();
    }
    return row;
  }

 private:
  static void verifyDelimiterCorrectness(const std::string& delimiter) {
    if (delimiter.empty()) {
      throw std::invalid_argument("Delimiter cannot be an empty string.");
    }
    // NOLINTNEXTLINE We don't use Abseil.
    if (delimiter.find('\\') != std::string::npos) {
      throw std::invalid_argument(
          "Delimiter cannot contain the escape character '\\'.");
    }
    // NOLINTNEXTLINE We don't use Abseil.
    if (delimiter.find('\n') != std::string::npos ||
        // NOLINTNEXTLINE We don't use Abseil.
        delimiter.find('\r') != std::string::npos) {
      throw std::invalid_argument(
          "Delimiter cannot contain the newline or return characters.");
    }
    if (delimiter == "\"") {
      throw std::invalid_argument(
          "Cannot use double quotes (\") as delimiter.");
    }
  }

  const std::string& _delimiter;
};

// I can add a unicode state
}  // namespace thirdai::dataset