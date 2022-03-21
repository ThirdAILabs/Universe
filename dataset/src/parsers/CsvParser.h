#pragma once

#include <cassert>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>

namespace thirdai::dataset {

template <typename VECTOR_T, typename LABEL_T>
class CsvParser {
 public:
  using VectorBuilder = std::function<VECTOR_T(const std::vector<float>&)>;
  using LabelBuilder = std::function<LABEL_T(uint32_t)>;

  CsvParser(VectorBuilder vb, LabelBuilder lb, char delimiter)
      : _vector_builder(vb), _label_builder(lb), _delimiter(delimiter) {}

  void parseBatch(uint32_t target_batch_size, std::ifstream& file,
                  std::vector<VECTOR_T>& vectors_out,
                  std::vector<LABEL_T>& labels_out) {
    uint32_t curr_batch_size = 0;
    uint32_t dim = 0;

    std::string line;
    while (curr_batch_size < target_batch_size && std::getline(file, line)) {
      if (line.empty()) {
        continue;
      }

      const char* start = line.c_str();
      const char* const line_end = line.c_str() + line.size();
      char* end;

      // Parse the label (first column of row in csv file).
      uint32_t label = std::strtoul(start, &end, 10);
      if (start == end) {
        throw std::invalid_argument(
            "Invalid dataset file: Found a line that doesn't start with a "
            "label.");
      }
      labels_out.push_back(_label_builder(label));
      if (line_end - end < 1) {
        throw std::invalid_argument(
            "Invalid dataset file: The line only contains a label.");
      }
      start = end;

      // Parse the vector itself (remaining columns of csv file).
      std::vector<float> values;
      while (start < line_end) {
        if (*start != _delimiter) {
          std::stringstream error_ss;
          error_ss << "Invalid dataset file: Found invalid character: "
                   << *start;
          throw std::invalid_argument(error_ss.str());
        }
        start++;
        if (start == line_end) {
          values.push_back(0);
        } else {
          float value = std::strtof(start, &end);
          if (start == end && *start != _delimiter) {
            std::stringstream error_ss;
            error_ss << "Invalid dataset file: Found invalid character: "
                     << *start;
            throw std::invalid_argument(error_ss.str());
          }
          // value defaults to 0, So if start == end but start == delimiter,
          // value = 0.
          values.push_back(value);
          start = end;
        }
      }

      if (dim != 0 && dim != values.size()) {
        throw std::invalid_argument(
            "Invalid dataset file: Contains different-dimensional vectors.\n");
      }

      dim = values.size();

      vectors_out.push_back(_vector_builder(values));

      curr_batch_size++;
    }
  }

 private:
  VectorBuilder _vector_builder;
  LabelBuilder _label_builder;
  char _delimiter;
};

}  // namespace thirdai::dataset
