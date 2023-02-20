#include "CsvParser.h"
#include <cstdint>
#include <exception>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>

namespace thirdai::dataset::parsers::CSV {

enum class ParserState {
  NewColumn,
  InQuotes,
  OutsideQuotes,
  EscapeInQuotes,
  EscapeOutsideQuotes,
  PotentialEndQuote,
  UnexpectedEOL,
};

// Declare parser helpers. Defined later so main parsing function
// isn't buried.

// Error handling helpers
static void validateDelimiter(char delimiter);
static void throwIfUnexpectedEOL(ParserState last_state,
                                 const std::string& line);
// Extracts the last column seen so far from `line`.
static std::string_view lastColumn(const std::string& line,
                                   ParserState prev_state, uint32_t start,
                                   uint32_t end);

static ParserState nextState(ParserState last_state, char current_char,
                             char delimiter);

// These functions define the state machine transitions.
namespace TransitionFrom {
static ParserState newColumn(char current_char, char delimiter);
static ParserState inQuotes(char current_char);
static ParserState outsideQuotes(char current_char, char delimiter);
static ParserState escapeInQuotes();
static ParserState escapeOutsideQuotes();
static ParserState potentialEndQuote(char current_char, char delimiter);
}  // namespace TransitionFrom

// Main parsing function.
std::vector<std::string_view> parseLine(const std::string& line,
                                        char delimiter) {
  validateDelimiter(delimiter);
  std::vector<std::string_view> parsed_columns;
  uint32_t column_start = 0;

  ParserState state = ParserState::NewColumn;
  uint32_t position = 0;
  for (char c : line) {
    auto prev_state = state;
    state = nextState(state, /* current_char= */ c, delimiter);
    if (state == ParserState::UnexpectedEOL) {
      break;
    }

    if (state == ParserState::NewColumn) {
      // If the new state is NewColumn, then we just finished parsing a column.
      parsed_columns.push_back(lastColumn(
          line, prev_state, /* start= */ column_start, /* end= */ position));
      column_start = position + 1;
    }

    position++;
  }

  throwIfUnexpectedEOL(state, line);

  parsed_columns.push_back(lastColumn(line, /* prev_state= */ state,
                                      /* start= */ column_start,
                                      /* end= */ position));

  return parsed_columns;
}

static void validateDelimiter(char delimiter) {
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

static void throwIfUnexpectedEOL(ParserState last_state,
                                 const std::string& line) {
  if (last_state == ParserState::EscapeInQuotes ||
      last_state == ParserState::EscapeOutsideQuotes ||
      last_state == ParserState::InQuotes ||
      last_state == ParserState::UnexpectedEOL) {
    throw std::invalid_argument(
        "Found unexpected newline, return character, or EOL in this line: \"" +
        line + "\"");
  }
}

static std::string_view lastColumn(const std::string& line,
                                   ParserState prev_state, uint32_t start,
                                   uint32_t end) {
  if (prev_state == ParserState::PotentialEndQuote) {
    start++;
    end--;
  }
  return {line.data() + start, end - start};
}

static ParserState nextState(ParserState last_state, char current_char,
                             char delimiter) {
  switch (last_state) {
    case ParserState::NewColumn:
      return TransitionFrom::newColumn(current_char, delimiter);
    case ParserState::InQuotes:
      return TransitionFrom::inQuotes(current_char);
    case ParserState::OutsideQuotes:
      return TransitionFrom::outsideQuotes(current_char, delimiter);
    case ParserState::EscapeInQuotes:
      return TransitionFrom::escapeInQuotes();
    case ParserState::EscapeOutsideQuotes:
      return TransitionFrom::escapeOutsideQuotes();
    case ParserState::PotentialEndQuote:
      return TransitionFrom::potentialEndQuote(current_char, delimiter);
      break;
    case ParserState::UnexpectedEOL:
      throw std::logic_error("Cannot transition from UnexpectedEOL");
    default:
      throw std::logic_error("Invalid parser state.");
  }
}

namespace TransitionFrom {

static ParserState newColumn(char current_char, char delimiter) {
  // Separate conditional since delimiter is not a constant.
  if (current_char == delimiter) {
    return ParserState::NewColumn;
  }

  switch (current_char) {
    case '"':
      return ParserState::InQuotes;
    case '\\':
      return ParserState::EscapeOutsideQuotes;
    case '\n':
    case '\r':
      return ParserState::UnexpectedEOL;
    default:
      return ParserState::OutsideQuotes;
  }
}

static ParserState inQuotes(char current_char) {
  switch (current_char) {
    case '"':
      return ParserState::PotentialEndQuote;
    case '\\':
      return ParserState::EscapeInQuotes;
    default:
      return ParserState::InQuotes;
  }
}

static ParserState outsideQuotes(char current_char, char delimiter) {
  if (current_char == delimiter) {
    return ParserState::NewColumn;
  }
  switch (current_char) {
    case '\\':
      return ParserState::EscapeOutsideQuotes;
    case '\n':
    case '\r':
      return ParserState::UnexpectedEOL;
    default:
      return ParserState::OutsideQuotes;
  }
}

static ParserState escapeInQuotes() { return ParserState::InQuotes; }

static ParserState escapeOutsideQuotes() { return ParserState::OutsideQuotes; }

static ParserState potentialEndQuote(char current_char, char delimiter) {
  if (current_char == delimiter) {
    return ParserState::NewColumn;
  }

  switch (current_char) {
    case '"':
      return ParserState::InQuotes;
    case '\\':
      return ParserState::EscapeOutsideQuotes;
    case '\n':
    case '\r':
      return ParserState::UnexpectedEOL;
    default:
      return ParserState::OutsideQuotes;
  }
}

}  // namespace TransitionFrom

}  // namespace thirdai::dataset::parsers::CSV