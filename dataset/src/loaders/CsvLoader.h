#include "LoaderInterface.h"
#include <fstream>
#include <string>
#include <string_view>
#include <iostream>

namespace thirdai::dataset {
    
class CsvLoader final : public Loader {
 public:
  CsvLoader(const std::string& filename, char delimiter, bool has_header)
      : _file(filename), _delimiter(delimiter), _has_header(has_header) {}

  std::optional<std::vector<std::string>> nextBatch(uint32_t batch_size) final {
    if (_file.eof()) {
      return std::nullopt;
    }

    std::vector<std::string> lines;
    std::string line;
    while (lines.size() < batch_size && std::getline(_file, line)) {
      lines.push_back(std::move(line));
    }
    return std::make_optional(std::move(lines));
  }

  void initialize() final {
    if (_has_header) {
      std::string line;
      std::getline(_file, line); // Get rid of header since we don't have a way to use it.
      // TODO(Geordie) How should we use the header?
    }
  }

  std::vector<std::string_view> parse(const std::string& line) final {
    std::vector<std::string_view> parsed;
    size_t start = 0;
    size_t end = 0;
    while (end != std::string::npos) {
      end = line.find(_delimiter, start);
      size_t len = end == std::string::npos ? line.size() - start : end - start;
      parsed.push_back(std::string_view(line.data() + start, len));
      start = end + 1;
    }
    return parsed;
  } 

 private:
  std::ifstream _file;
  char _delimiter;
  bool _has_header;
};

} // namespace thirdai::dataset

