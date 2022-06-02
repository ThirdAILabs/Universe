#pragma once

#include <cassert>
#include <fstream>
#include <functional>
#include <vector>

namespace thirdai::dataset {

template <typename VECTOR_T, typename LABEL_T>
class SvmParser {
 public:
  using VectorBuilder = std::function<VECTOR_T(const std::vector<uint32_t>&,
                                               const std::vector<float>&)>;
  using LabelBuilder = std::function<LABEL_T(const std::vector<uint32_t>&)>;

  SvmParser(VectorBuilder vb, LabelBuilder lb)
      : _vector_builder(vb), _label_builder(lb) {}

  uint32_t parseBatch(uint32_t target_batch_size, std::ifstream& file,
                  std::vector<VECTOR_T>& vectors_out,
                  std::vector<LABEL_T>& labels_out) {
    uint32_t curr_batch_size = 0;
    uint32_t max_index = 0;
    std::string line;
    while (curr_batch_size < target_batch_size && std::getline(file, line)) {
      const char* start = line.c_str();
      const char* const line_end = line.c_str() + line.size();
      char* end;

      // Parse the labels. The labels are comma separated without spaces.
      // Ex: 3,4,13
      std::vector<uint32_t> labels;
      do {
        uint32_t label = std::strtoul(start, &end, 10);
        labels.push_back(label);
        start = end;
      } while ((*start++) == ',');
      labels_out.push_back(this->_label_builder(labels));

      // Parse the vector itself. The elements are given in <index>:<value>
      // pairs with tabs or spaces between each pair. There should also be a
      // tab/space between the labels and first pair.
      std::vector<uint32_t> indices;
      std::vector<float> values;
      do {
        uint32_t index = std::strtoul(start, &end, 10);
        start = end + 1;
        float value = std::strtof(start, &end);
        indices.push_back(index);
        values.push_back(value);
        max_index = std::max(max_index, index);
        start = end;

        while ((*start == ' ' || *start == '\t') && start < line_end) {
          start++;
        }
      } while (*start != '\n' && start < line_end);

      vectors_out.push_back(this->_vector_builder(indices, values));

      curr_batch_size++;
    }
    return max_index;
  }

 private:
  VectorBuilder _vector_builder;
  LabelBuilder _label_builder;
};

}  // namespace thirdai::dataset
