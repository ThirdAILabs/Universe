#include "CsvParser.h"
#include <cstdint>
#include <exception>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>

namespace thirdai::dataset::parsers::CSV {

StateMachine::StateMachine(char delimiter)
    : _delimiter(delimiter), _state(ParserState::NewLine) {
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
    case ParserState::EscapeInQuotes:
      // The character after the escape character is treated like a regular
      // character.
      _state = ParserState::RegularInQuotes;
      break;
    case ParserState::UnescapedDelimiterInQuotes:
      // UnescapedDelimiterInQutoes is a special case of InQuotes with the same
      // out-transitions.
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

void StateMachine::setState(ParserState state) { _state = state; }

void StateMachine::setPreviousState(ParserState state) {
  _previous_state = state;
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
    return ParserState::UnescapedDelimiterInQuotes;
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
static std::string_view columnView(const std::string& line,
                                   ParserState column_end_state,
                                   uint32_t start_index, uint32_t end_index) {
  if (column_end_state == ParserState::PotentialEndQuote) {
    // If the previous state is PotentialEndQuote, then the previous column
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
static bool malformedQuotedColumn(StateMachine& state_machine,
                                  bool is_last_char) {
  auto regular_char_after_end_quote =
      state_machine.previousState() == ParserState::PotentialEndQuote &&
      state_machine.state() == ParserState::RegularOutsideQuotes;
  auto still_in_quotes =
      state_machine.state() == ParserState::EscapeInQuotes ||
      state_machine.state() == ParserState::RegularInQuotes ||
      state_machine.state() == ParserState::UnescapedDelimiterInQuotes;
  return (is_last_char && still_in_quotes) || regular_char_after_end_quote;
}

std::vector<std::string_view> parseLine(const std::string& line,
                                        char delimiter) {
  std::vector<std::string_view> parsed_columns;

  StateMachine state_machine(delimiter);
  uint32_t column_start = 0;
  std::optional<uint32_t> first_delimiter_in_quotes;
  for (uint32_t position = 0; position < line.size(); position++) {
    state_machine.transition(line[position]);

    // Side effects of state transition.

    if (state_machine.state() == ParserState::NewLine) {
      throw std::invalid_argument("Found unexpected unquoted newline in: \"" +
                                  line + "\"");
    }

    if (state_machine.state() == ParserState::UnescapedDelimiterInQuotes &&
        !first_delimiter_in_quotes) {
      first_delimiter_in_quotes = position;
    }

    /*
      If quoted column is malformed and we have seen delimiters inside the
      quotes, reset position to first delimiter in quotes and treat it like the
      end of the column.
    */
    if (first_delimiter_in_quotes &&
        malformedQuotedColumn(
            /* state_machine= */ state_machine,
            /* is_last_char= */ position == line.size() - 1)) {
      position = *first_delimiter_in_quotes;
      state_machine.setState(ParserState::NewColumn);
      state_machine.setPreviousState(ParserState::RegularOutsideQuotes);
    }

    if (state_machine.state() == ParserState::NewColumn) {
      // If the new state is NewColumn, then we just finished parsing a
      // column.
      parsed_columns.push_back(columnView(
          line, /* column_end_state= */ state_machine.previousState(),
          /* start_index= */ column_start, /* end_index= */ position));
      column_start = position + 1;
      first_delimiter_in_quotes = {};
    }
  }

  parsed_columns.push_back(
      columnView(line, /* column_end_state= */ state_machine.state(),
                 /* start_index= */ column_start,
                 /* end_index= */ line.size()));

  return parsed_columns;
}

}  // namespace thirdai::dataset::parsers::CSV