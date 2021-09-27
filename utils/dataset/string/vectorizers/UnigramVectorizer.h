#include "StringVectorizer.h"

namespace thirdai::utils {

class UnigramVectorizer : public StringVectorizer {
 private:
  /* data */
  u_int32_t _murmur_seed = 42;
  VECTOR_TYPE _vector_type;
 public:
  //UnigramVectorizer(GlobalFreq* globalFreq) : StringVectorizer(globalFreq){}
  UnigramVectorizer(uint32_t start_idx, uint32_t max_dim);
  virtual void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                         std::vector<float>& values, VECTOR_TYPE vector_type);
};

}  // namespace thirdai::utils