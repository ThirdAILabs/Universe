#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace thirdai::dataset::parsers::CSV {

enum class ParserState {
  NewLine,
  NewColumn,
  RegularInQuotes,
  UnescapedDelimiterInQuotes,
  RegularOutsideQuotes,
  EscapeInQuotes,
  EscapeOutsideQuotes,
  PotentialEndQuote,
};

class StateMachine {
 public:
  explicit StateMachine(char delimiter);

  void transition(char current_char);

  ParserState state() const;

  ParserState previousState() const;

  void setState(ParserState);

  void setPreviousState(ParserState);

 private:
  char _delimiter;
  ParserState _state;
  ParserState _previous_state;

  static void validateDelimiter(char delimiter);
  ParserState fromNewColumn(char current_char) const;
  ParserState fromRegularInQuotes(char current_char) const;
  ParserState fromRegularOutsideQuotes(char current_char) const;
  ParserState fromPotentialEndQuote(char current_char) const;
};

/**
 * Parses a CSV line. Expects a single line with no unescaped or unquoted
 * newline character. This is the main parsing function.
 */
std::vector<std::string_view> parseLine(const std::string& line,
                                        char delimiter);

}  // namespace thirdai::dataset::parsers::CSV