#include "../Factory.h"
#include "../batch_types/SparseBatch.h"
#include "global_frequency/GlobalFreq.h"
#include "loaders/StringLoader.h"
#include "vectorizers/CompositeVectorizer.h"

namespace thirdai::utils {

template <typename Loader_t>
class StringFactory : public Factory<SparseBatch> {
 private:
  Loader_t _loader;
  CompositeVectorizer _vectorizer;
  std::unordered_map<uint32_t, float> _idfMap;

 public:
  // Without TF-IDF
  explicit StringFactory(vectorizer_config_t vectorizer_config)
      : _loader(),
        _vectorizer(std::move(vectorizer_config)),
        _idfMap(std::move(std::unordered_map<uint32_t, float>())) {}

  // With global frequencies object for TF-IDF
  explicit StringFactory(vectorizer_config_t vectorizer_config,
                         GlobalFreq<Loader_t>& global_freq)
      : _loader(),
        _vectorizer(std::move(vectorizer_config)),
        _idfMap(global_freq._idfMap) {
    if (_vectorizer.getConfigHash() != global_freq.getVectorizerConfigHash()) {
      std::invalid_argument(
          "Invalid global frequencies object; vectorizer configurations do not "
          "match.");
    }
  }

  SparseBatch parse(std::ifstream& file, uint32_t target_batch_size,
                    uint64_t start_id) override {
    std::vector<std::string> strings;
    _loader.loadStrings(file, target_batch_size, strings);
    uint32_t actual_batch_size = strings.size();

    std::vector<SparseVector> vectors;
    vectors.resize(actual_batch_size);

    std::vector<std::vector<uint32_t>> labels;
    labels.resize(actual_batch_size);

    for (uint32_t v = 0; v < actual_batch_size; v++) {
      std::vector<uint32_t> indices;
      std::vector<float> values;
      _vectorizer.vectorize(strings[v], indices, values, _idfMap);
      SparseVector string_vec(indices.size());
      string_vec.indices = indices.data();
      string_vec.values = values.data();
      vectors[v] = string_vec;
      labels[v].push_back(0);
    }

    return SparseBatch(std::move(vectors), std::move(labels), start_id);
  }
};

}  // namespace thirdai::utils