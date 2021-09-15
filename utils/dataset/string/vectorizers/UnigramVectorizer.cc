#include "UnigramVectorizer.h"

namespace thirdai::utils {

UnigramVectorizer::~UnigramVectorizer() {}

void UnigramVectorizer::vectorize(const std::string& str,
                                  std::vector<uint32_t>& indices,
                                  std::vector<float>& values,
                                  VECTOR_TYPE vector_type) {
  switch (vector_type) {
    case VECTOR_TYPE::MURMUR:
      /* code */
      break;
    case VECTOR_TYPE::TFIDF:
      indices.clear();
      values.clear();
      for (auto token : str) {
        int idf = _globalFreq->getIdf(token);
        int tf = _globalFreq->getTF(token, str);
        int tokenID = _globalFreq->getTokenID(token);
        indices.push_back(tokenID);
        values.push_back(idf * tf);
        // TODO: What about markers?
      }
      break;

    default:
      indices.clear();
      values.clear();
      for (auto token : str) {
        int idf = _globalFreq->getIdf(token);
        int tf = _globalFreq->getTF(token, str);
        int tokenID = _globalFreq->getTokenID(token);
        indices.push_back(tokenID);
        values.push_back(idf * tf);
        // TODO: What about markers?
      }
      break;
  }
}
}  // namespace thirdai::utils%        