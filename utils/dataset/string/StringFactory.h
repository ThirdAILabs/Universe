#include "../Factory.h"
#include "../batch_types/SparseBatch.h"
#include "global_frequency/GlobalFreq.h"
#include "loaders/StringLoader.h"
#include "vectorizers/CompositeVectorizer.h"

namespace thirdai::utils::dataset {

template <typename LOADER_T>
class StringFactory : public Factory<SparseBatch> {
 private:
  static std::unordered_map<uint32_t, float> _default_empty_map;
  LOADER_T _loader;
  CompositeVectorizer _vectorizer;
  std::unordered_map<uint32_t, float>& _idf_map;

 public:
  // Without TF-IDF
  explicit StringFactory(vectorizer_config_t& vectorizer_config)
      : _loader(),
        _vectorizer(vectorizer_config),  // Forces a copy so it can be reused
                                         // outside this instance.
        _idf_map(_default_empty_map) {}

  // With global frequencies object for TF-IDF
  // Does not need to take in separate vectorizer config since it can be
  // retrieved from global_freq, and the configurations must be the same
  explicit StringFactory(const GlobalFreq<LOADER_T>& global_freq)
      : _loader(),
        _vectorizer(std::move(global_freq.getVectorizerConfig())),
        _idf_map(global_freq.getIdfMap()) {}

  SparseBatch parse(std::ifstream& file, uint32_t target_batch_size,
                    uint64_t start_id) override {
    std::vector<std::string> strings;
    std::vector<std::vector<uint32_t>> labels;
    _loader.loadStringsAndLabels(file, target_batch_size, strings, labels);
    uint32_t actual_batch_size = strings.size();

    std::vector<SparseVector> vectors(actual_batch_size);

    for (uint32_t v = 0; v < actual_batch_size; v++) {
      std::vector<uint32_t> indices;
      std::vector<float> values;
      _vectorizer.vectorize(strings[v], indices, values, _idf_map);

      SparseVector string_vec(indices.size());

      // Need to copy each element as the vectors will go out of scope and
      // deallocate internal array.
      string_vec.indices = new uint32_t[indices.size()];
      std::copy(indices.begin(), indices.end(), string_vec.indices);
      string_vec.values = new float[values.size()];
      std::copy(values.begin(), values.end(), string_vec.values);

      vectors[v] = string_vec;
    }

    return {std::move(vectors), std::move(labels), start_id};
  }
};

}  // namespace thirdai::utils::dataset