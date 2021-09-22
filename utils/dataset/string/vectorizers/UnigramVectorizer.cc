#include "UnigramVectorizer.h"

namespace thirdai::utils {

UnigramVectorizer::~UnigramVectorizer() {}

virtual void UnigramVectorizer::vectorize(const std::string& str,
                                          std::vector<uint32_t>& indices,
                                          std::vector<float>& values,
                                          VECTOR_TYPE vector_type) {
    for (auto token : str) {
      // TODO: make the frequency count similar to trigram
      int idf = _globalFreq->getIdf(token);
      int tf = _globalFreq->getTF(token, str);
      int tokenID = _globalFreq->getTokenID(token);
      indices.push_back(tokenID);
      values.push_back(idf * tf);
      // TODO: What about markers?
    }
  }

}  // namespace thirdai::utils