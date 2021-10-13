#include "../batch_types/SparseBatch.h"
#include "vectorizers/StringVectorizer.h"
#include "loaders/StringLoader.h"

namespace thirdai::utils {

class StringParser {
  static SparseBatch parse(std::ifstream& file, uint32_t target_batch_size,
                       uint64_t start_id, StringLoader *loader, StringVectorizer *vectorizer) {

    std::vector<std::string> strings;
    loader->loadStrings(file, target_batch_size, strings);
    uint32_t actual_batch_size = strings.size();

    std::vector<SparseVector> vectors;
    vectors.resize(actual_batch_size);

    std::vector<std::vector<uint32_t>> labels;
    labels.resize(actual_batch_size);
    
    for (uint32_t v = 0; v < actual_batch_size; v++) {
      std::vector<uint32_t> indices;
      std::vector<float> values;
      vectorizer->vectorize(strings[v], indices, values);
      SparseVector string_vec(indices.size());
      string_vec.indices = indices.data();
      string_vec.values = values.data();
      vectors[v] = string_vec;
      labels[v].push_back(0);
    }
    
    return SparseBatch(std::move(vectors), std::move(labels), start_id);
  }
};

} // namespace thirdai::utils