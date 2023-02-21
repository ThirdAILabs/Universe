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

  void setState(ParserState);

 private:
  char _delimiter;
  ParserState _state;

  static void validateDelimiter(char delimiter);
  ParserState fromNewColumn(char current_char) const;
  ParserState fromRegularInQuotes(char current_char) const;
  ParserState fromRegularOutsideQuotes(char current_char) const;
  ParserState fromPotentialEndQuote(char current_char) const;
};

/**
 * Parses a CSV line. Expects a single line with no unescaped or unquoted
 * newline character.
 */
std::vector<std::string_view> parseLine(const std::string& line,
                                        char delimiter);

}  // namespace thirdai::dataset::parsers::CSV