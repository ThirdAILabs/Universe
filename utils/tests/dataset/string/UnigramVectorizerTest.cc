#include "../../../dataset/string/vectorizers/UnigramVectorizer.h"
#include <gtest/gtest.h>

using std::cout;
using std::endl;

namespace thirdai::utils {

std::string simple =
    "this is a string to test unigram vectorizer string string unigram";
u_int32_t start_idx = 0;
u_int32_t max_dim = 100000;
std::vector<u_int32_t> indices;
std::vector<float> values;

TEST(UnigramVectorizerTest, Vectorize) {
  UnigramVectorizer unigram_vectorizer(start_idx, max_dim, VALUE_TYPE::TF);
  unigram_vectorizer.vectorize(simple, indices, values);
  ASSERT_EQ(indices.size(), values.size());
  cout << "indices: ";
  for (auto i : indices) {
    cout << i << " ";
  }
  cout << endl;
  cout << "values: ";
  for (auto i : values) {
    cout << i << " ";
  }
  cout << endl;
}
}  // namespace thirdai::utils