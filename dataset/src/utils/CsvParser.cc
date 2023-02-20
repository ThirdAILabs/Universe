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

namespace TransitionFrom {
static ParserState newColumn(char current_char, char delimiter);
static ParserState inQuotes(char current_char);
static ParserState outsideQuotes(char current_char, char delimiter);
static ParserState escapeInQuotes();
static ParserState escapeOutsideQuotes();
static ParserState potentialEndQuote(char current_char, char delimiter);
}  // namespace TransitionFrom

static void validateDelimiter(char delimiter);

static std::string_view lastColumn(const std::string& row, uint32_t start,
                                   uint32_t end, ParserState prev_state);

static void throwIfUnexpectedEOL(ParserState last_state,
                                 const std::string& line);

std::vector<std::string_view> parseLine(const std::string& line,
                                        char delimiter) {
  validateDelimiter(delimiter);
  std::vector<std::string_view> parsed_columns;
  uint32_t column_start = 0;

  ParserState state = ParserState::NewColumn;
  uint32_t position = 0;
  for (char c : line) {
    auto prev_state = state;
    switch (state) {
      case ParserState::NewColumn:
        state = TransitionFrom::newColumn(/* current_char= */ c,
                                          /* delimiter= */ delimiter);
        break;
      case ParserState::InQuotes:
        state = TransitionFrom::inQuotes(/* current_char= */ c);
        break;
      case ParserState::OutsideQuotes:
        state = TransitionFrom::outsideQuotes(/* current_char= */ c,
                                              /* delimiter= */ delimiter);
        break;
      case ParserState::EscapeInQuotes:
        state = TransitionFrom::escapeInQuotes();
        break;
      case ParserState::EscapeOutsideQuotes:
        state = TransitionFrom::escapeOutsideQuotes();
        break;
      case ParserState::PotentialEndQuote:
        state = TransitionFrom::potentialEndQuote(/* current_char= */ c,
                                                  /* delimiter= */ delimiter);
        break;
      case ParserState::UnexpectedEOL:
        throw std::logic_error("Cannot transition from UnexpectedEOL");
        break;
      default:
        throw std::logic_error("Invalid parser state.");
    }

    if (state == ParserState::UnexpectedEOL) {
      break;
    }

    // If the new state is NewColumn, then we just finished parsing a column.
    if (state == ParserState::NewColumn) {
      parsed_columns.push_back(
          lastColumn(line, column_start, position, prev_state));
      column_start = position + 1;
    }

    position++;
  }

  throwIfUnexpectedEOL(state, line);

  parsed_columns.push_back(lastColumn(line, column_start, position, state));

  return parsed_columns;
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

static std::string_view lastColumn(const std::string& row, uint32_t start,
                                   uint32_t end, ParserState prev_state) {
  if (prev_state == ParserState::PotentialEndQuote) {
    start++;
    end--;
  }
  return {row.data() + start, end - start};
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

}  // namespace thirdai::dataset::parsers::CSV