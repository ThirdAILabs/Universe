#include "DataSource.h"
#include <dataset/src/utils/CsvParser.h>
#include <utils/StringManipulation.h>
#include <iostream>
#include <stdexcept>

namespace thirdai::dataset {

std::optional<std::string> CsvDataSource::nextLine() {
  std::cout << "Called nextLine" << std::endl;
  parsers::CSV::StateMachine state_machine(_delimiter);
  std::vector<std::string> buffer;
  while (auto line = _source->nextLine()) {
    std::cout << "raw line '" << *line << "'" << std::endl;
    buffer.push_back(*line);
    if (!inQuotesAtEndOfLine(state_machine, *line)) {
      std::cout << "break because not in quotes at end of line" << std::endl;
      break;
    }
  }
  std::cout << "out of while loop" << std::endl;
  if (inQuotes(state_machine.state())) {
    throw std::invalid_argument("Reached EOF without closing quoted column.");
  }
  if (buffer.empty()) {
    return std::nullopt;
  }
  auto line = text::join(buffer, "\n");
  std::cout << "CSVDataSource: returning line '" << line << "'" << std::endl;
  std::cout << "Line size: " << line.size() << std::endl;

  return line;
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

bool CsvDataSource::inQuotesAtEndOfLine(
    parsers::CSV::StateMachine& state_machine, const std::string& line) {
  for (char c : line) {
    state_machine.transition(c);
  }
  return inQuotes(state_machine.state());
}
}  // namespace thirdai::dataset