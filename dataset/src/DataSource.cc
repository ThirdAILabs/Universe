#include "DataSource.h"
#include <dataset/src/utils/CsvParser.h>
#include <utils/StringManipulation.h>
#include <iostream>
#include <stdexcept>

namespace thirdai::dataset {

std::optional<std::string> CsvDataSource::nextLine() {
  parsers::CSV::StateMachine state_machine(_delimiter);
  std::vector<std::string> buffer;
  std::optional<uint32_t> newline_position;
  while (!newline_position) {
    if (auto line = nextRawLine()) {
      newline_position = findNewline(state_machine, *line);
      if (!newline_position) {
        buffer.push_back(*line);
      } else {
        if (*newline_position == line->size()) {
          buffer.push_back(*line);
        } else if (*newline_position == line->size() - 1) {
          buffer.push_back(line->substr(0, *newline_position));
        } else {
          buffer.push_back(line->substr(0, *newline_position));
          _remains = line->substr(*newline_position);
        }
      }
    } else {
      switch (state_machine.state()) {
        case parsers::CSV::ParserState::DelimiterInQuotes:
        case parsers::CSV::ParserState::EscapeInQuotes:
        case parsers::CSV::ParserState::RegularInQuotes:
          throw std::invalid_argument(
              "Reached EOF without closing quoted column.");
        default:
          break;
      }
      break;
    }
  }
  if (buffer.empty()) {
    return std::nullopt;
  }
  return text::join(buffer, "\n");
}

std::optional<std::vector<std::string>> CsvDataSource::nextBatch(
    size_t target_batch_size) {
  std::vector<std::string> lines;
  while (lines.size() < target_batch_size) {
    if (auto next_line = nextLine()) {
      lines.push_back(*next_line);
    } else {
      break;
    }
  }

  if (lines.empty()) {
    return std::nullopt;
  }

  return std::make_optional(std::move(lines));
}
std::optional<uint32_t> CsvDataSource::findNewline(
    parsers::CSV::StateMachine& state_machine, const std::string& line) {
  for (uint32_t position = 0; position < line.size(); position++) {
    state_machine.transition(line[position]);
    if (state_machine.state() == parsers::CSV::ParserState::NewLine) {
      return position;
    }
  }
  state_machine.transition('\n');
  if (state_machine.state() == parsers::CSV::ParserState::NewLine) {
    return line.size();
  }
  return std::nullopt;
}
std::optional<std::string> CsvDataSource::nextRawLine() {
  if (_remains) {
    auto line = std::move(_remains);
    _remains = {};
    return line;
  }
  return _source->nextLine();
}
}  // namespace thirdai::dataset