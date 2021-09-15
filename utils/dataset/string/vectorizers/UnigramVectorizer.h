#include "StringVectorizer.h"

namespace thirdai::utils {

class UnigramVectorizer : public StringVectorizer {
 private:
  /* data */
 public:
  UnigramVectorizer(GlobalFreq* globalFreq) : StringVectorizer(globalFreq);
  virtual void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                         std::vector<float>& values, VECTOR_TYPE vector_type);
};

}  // namespace thirdai::utils%            