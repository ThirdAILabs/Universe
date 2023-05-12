#pragma once

#include <string>
#include <vector>

namespace thirdai::dataset::parsers::CSV {

enum class ParserState {
  None,
  NewLine,
  NewColumn,
  RegularInQuotes,
  DelimiterInQuotes,
  RegularOutsideQuotes,
  EscapeInQuotes,
  EscapeOutsideQuotes,
  PotentialEndQuote,
};

/**
 * The state machine defines transitions between ParserStates and stores the
 * current and previous states.
 *
 * This object is meant to aid left-to-right parsing of a CSV file or line.
 * It is only concerned with the state of parsing and does not handle side
 * effects such as splitting a string into column substrings or throwing use
 * case-specific errors.
 */
class StateMachine {
 public:
  explicit StateMachine(char delimiter);

  void transition(char current_char);

  ParserState state() const;

  ParserState previousState() const;

  void setState(ParserState current_state, ParserState previous_state);

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
 * Parses a CSV line. Expects a single line with no unquoted
 * newline character. Throws an error if it sees an unquoted newline character
 * in the middle of the line, and trims the character if it is at the end of the
 * line. This is the main parsing function.
 */
std::vector<std::string> parseLine(const std::string& line, char delimiter);

}  // namespace thirdai::dataset::parsers::CSV