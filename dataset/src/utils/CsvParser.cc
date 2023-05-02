#include "CsvParser.h"
#include <cstdint>
#include <exception>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>

namespace thirdai::dataset::parsers::CSV {

StateMachine::StateMachine(char delimiter)
    : _delimiter(delimiter),
      _state(ParserState::NewLine),
      _previous_state(ParserState::None) {
  validateDelimiter(_delimiter);
}

void StateMachine::transition(char current_char) {
  _previous_state = _state;

  switch (_state) {
    case ParserState::NewLine:
      // NewLine is a special case of NewColumn with the same out-transitions.
    case ParserState::NewColumn:
      _state = fromNewColumn(current_char);
      break;
    // EscapeInQuotes and DelimiterInQutoes and are special cases of
    // RegularInQuotes with the same out-transitions.
    case ParserState::EscapeInQuotes:
    case ParserState::DelimiterInQuotes:
    case ParserState::RegularInQuotes:
      _state = fromRegularInQuotes(current_char);
      break;
    case ParserState::EscapeOutsideQuotes:
      // EscapeOutsideQutoes is a special case of RegularOutsideQuotes with the
      // same out-transitions
    case ParserState::RegularOutsideQuotes:
      _state = fromRegularOutsideQuotes(current_char);
      break;
    case ParserState::PotentialEndQuote:
      _state = fromPotentialEndQuote(current_char);
      break;
    default:
      throw std::logic_error("Invalid parser state.");
  }
}

ParserState StateMachine::state() const { return _state; }

ParserState StateMachine::previousState() const { return _previous_state; }

void StateMachine::setState(ParserState current_state,
                            ParserState previous_state) {
  _state = current_state;
  _previous_state = previous_state;
}

void StateMachine::validateDelimiter(char delimiter) {
  switch (delimiter) {
    case '\\':
      throw std::invalid_argument(
          "Cannot use escape character '\\' as delimiter.");
    case '\n':
      throw std::invalid_argument("Cannot use newline character as delimiter.");
    case '\r':
      throw std::invalid_argument("Cannot use return character as delimiter.");
    case '\"':
      throw std::invalid_argument(
          "Cannot use double quotes '\"' as delimiter.");
    default:
      break;
  }
}

ParserState StateMachine::fromNewColumn(char current_char) const {
  // Not checked in switch statement because delimiter is not a constant
  if (current_char == _delimiter) {
    return ParserState::NewColumn;
  }

  switch (current_char) {
    case '"':
      return ParserState::RegularInQuotes;
    case '\\':
      return ParserState::EscapeOutsideQuotes;
    case '\n':
      return ParserState::NewLine;
    default:
      return ParserState::RegularOutsideQuotes;
  }
}

ParserState StateMachine::fromRegularInQuotes(char current_char) const {
  // Not checked in switch statement because delimiter is not a constant
  if (current_char == _delimiter) {
    return ParserState::DelimiterInQuotes;
  }

  switch (current_char) {
    case '"':
      return ParserState::PotentialEndQuote;
    case '\\':
      return ParserState::EscapeInQuotes;
    default:
      return ParserState::RegularInQuotes;
  }
}

ParserState StateMachine::fromRegularOutsideQuotes(char current_char) const {
  // Not checked in switch statement because delimiter is not a constant
  if (current_char == _delimiter) {
    return ParserState::NewColumn;
  }

  switch (current_char) {
    case '\\':
      return ParserState::EscapeOutsideQuotes;
    case '\n':
      return ParserState::NewLine;
    default:
      return ParserState::RegularOutsideQuotes;
  }
}

ParserState StateMachine::fromPotentialEndQuote(char current_char) const {
  // Not checked in switch statement because delimiter is not a constant
  if (current_char == _delimiter) {
    return ParserState::NewColumn;
  }

  switch (current_char) {
    /*
      CSV standard: Two double quotes inside quoted string are treated like
      escaped double quotes. E.g. "I just saw ""Dear Evan Hansen"", 10/10!"
      is interpreted as:
      I just saw "Dear Evan Hansen", 10/10!
    */
    case '"':
      return ParserState::RegularInQuotes;
    // In all the other cases, since we've only seen one double quotation mark,
    // we treat it as the end quote, thus we are now "outside quotes".
    case '\\':
      return ParserState::EscapeOutsideQuotes;
    case '\n':
      return ParserState::NewLine;
    default:
      return ParserState::RegularOutsideQuotes;
  }
}

// Extracts the last column seen so far from `line`.
static std::string columnView(const std::string& line,
                              ParserState column_end_state,
                              uint32_t start_index, uint32_t end_index) {
  if (column_end_state == ParserState::PotentialEndQuote) {
    // If the column end state is PotentialEndQuote, then the previous column
    // must be quoted. Thus, we trim the quotes by incrementing start and
    // decrementing end.
    start_index++;
    end_index--;
  }
  return {line.data() + start_index, end_index - start_index};
}

/**
 * Quoted column is malformed if we reach end of line without seeing an end
 * quote or if an end quote is followed by a regular character.
 */
static bool quotesAreMalformed(StateMachine& state_machine, bool is_last_char) {
  auto regular_char_after_end_quote =
      state_machine.previousState() == ParserState::PotentialEndQuote &&
      state_machine.state() == ParserState::RegularOutsideQuotes;
  auto still_in_quotes =
      state_machine.state() == ParserState::EscapeInQuotes ||
      state_machine.state() == ParserState::RegularInQuotes ||
      state_machine.state() == ParserState::DelimiterInQuotes;
  return (is_last_char && still_in_quotes) || regular_char_after_end_quote;
}

static std::string trimNewlineAtEndOfLine(const std::string& line) {
  if (line.size() >= 2 && line.substr(line.size() - 2) == "\r\n") {
    return {line.data(), line.size() - 2};
  }
  if (!line.empty() && line.back() == '\r') {
    return {line.data(), line.size() - 1};
  }
  if (!line.empty() && line.back() == '\n') {
    return {line.data(), line.size() - 1};
  }
  return {line.data(), line.size()};
}

/**
 * Theoretically, one can reset an optional x by doing any of the following:
 * x = {};
 * x.reset();
 * x = std::nullopt;
 * x = std::optional<T>();
 * x = std::optional<T>{};
 * But for some reason, all of these cause either the blade compiler or the CI
 * compiler to complain. Seems to be a g++ bug. So we'll have to settle with
 * this for now. Abstracted into a function to avoid bloating the main function
 * with this comment.
 */
static void resetOptional(std::optional<uint32_t>& optional) {
  std::optional<uint32_t> opt;
  optional = opt;
}

std::vector<std::string> parseLine(const std::string& untrimmed_line,
                                   char delimiter) {
  std::vector<std::string> parsed_columns;

  auto line = trimNewlineAtEndOfLine(untrimmed_line);

  StateMachine state_machine(delimiter);
  uint32_t column_start = 0;
  std::optional<uint32_t> first_delimiter_in_quotes;
  for (uint32_t position = 0; position < line.size(); position++) {
    state_machine.transition(line[position]);

    bool is_last_char = position == line.size() - 1;

    // Side effects of state transition.

    if (state_machine.state() == ParserState::NewLine && !is_last_char) {
      throw std::invalid_argument("Found unexpected unquoted newline in: \"" +
                                  std::string(line) + "\"");
    }

    if (state_machine.state() == ParserState::DelimiterInQuotes &&
        !first_delimiter_in_quotes) {
      first_delimiter_in_quotes = position;
    }

    if (first_delimiter_in_quotes &&
        quotesAreMalformed(state_machine, is_last_char)) {
      position = *first_delimiter_in_quotes;
      state_machine.setState(
          /* current_state= */ ParserState::NewColumn,
          /* previous_state= */ ParserState::RegularOutsideQuotes);
    }

    if (state_machine.state() == ParserState::NewColumn) {
      // If the new state is NewColumn, then we just finished parsing a
      // column.
      parsed_columns.push_back(columnView(
          line, /* column_end_state= */ state_machine.previousState(),
          /* start_index= */ column_start, /* end_index= */ position));
      column_start = position + 1;
      resetOptional(first_delimiter_in_quotes);
    }
  }

  parsed_columns.push_back(
      columnView(line, /* column_end_state= */ state_machine.state(),
                 /* start_index= */ column_start,
                 /* end_index= */ line.size()));

  return parsed_columns;
}

}  // namespace thirdai::dataset::parsers::CSV