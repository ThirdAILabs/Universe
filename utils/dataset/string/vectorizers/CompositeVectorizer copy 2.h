#include "StringVectorizer.h"
#include "TriGramVectorizer.h"
#include <limits>
#include <vector>
#include <initializer_list>
#include <utility>

namespace thirdai::utils {
    class CompositeVectorizer : public StringVectorizer {
        public:
        CompositeVectorizer(std::initializer_list<StringVectorizer*> vectorizers): StringVectorizer(0, std::numeric_limits<uint32_t>::max()) {
            _dim = 0;
            _num_vectorizers = 0;
            _vectorizers = new StringVectorizer*[vectorizers.size()];
            for (const auto& v_ptr : vectorizers) {
                _vectorizers[_num_vectorizers] = v_ptr;
                _dim += v_ptr->getDimension();
                _num_vectorizers++;
            }
        }

        void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                         std::vector<float>& values) override {
            for (size_t i = 0; i < _num_vectorizers; i++) {
                _vectorizers[i]->vectorize(str, indices, values);
            }
        }

        virtual ~CompositeVectorizer(){
            delete[] _vectorizers;
        };

        private: 
        uint32_t _num_vectorizers;
        StringVectorizer** _vectorizers;
    };

} // namespace thirdai::utils
