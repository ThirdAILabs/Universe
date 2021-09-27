#include "StringVectorizer.h"
#include "TriGramVectorizer.h"
#include <limits>
#include <vector>

namespace thirdai::utils {
enum class VECTORIZER { CHAR_TRIGRAM, WORD_UNIGRAM };
class CompositeVectorizer : public StringVectorizer {
 public:
  CompositeVectorizer()
      : StringVectorizer(0, std::numeric_limits<uint32_t>::max()) {
    _dim = 0;
  }

  void addVectorizer(VECTORIZER vectorizer, uint32_t max_dim) {
    StringVectorizer* v_ptr;
    switch (vectorizer) {
      case VECTORIZER::CHAR_TRIGRAM:
        v_ptr = new TriGramVectorizer(_dim, max_dim);
        break;

      default:
        break;
    }
    _vectorizers.push_back(v_ptr);
    _dim += v_ptr->getDimension();
  }

  void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                 std::vector<float>& values) override {
    for (auto& vectorizer : _vectorizers) {
      vectorizer->vectorize(str, indices, values);
    }
  }

  virtual ~CompositeVectorizer() {
    for (auto& v_ptr : _vectorizers) {
      delete v_ptr;
    }
  };

 private:
  std::vector<StringVectorizer*> _vectorizers;
};

}  // namespace thirdai::utils
