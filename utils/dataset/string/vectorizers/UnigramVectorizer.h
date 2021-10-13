#include "../../../hashing/MurmurHash.h"
#include "StringVectorizer.h"
#include <sstream>

namespace thirdai::utils {

class UnigramVectorizer : public StringVectorizer {
 private:
  /* data */
  u_int32_t _murmur_seed = 4242;
  VALUE_TYPE _value_type;

 public:
  // UnigramVectorizer(GlobalFreq* globalFreq) : StringVectorizer(globalFreq){}
  UnigramVectorizer(uint32_t start_idx, uint32_t max_dim,
                    VALUE_TYPE value_type);
  void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                 std::vector<float>& values) override;

  void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                 std::vector<float>& values,
                 const std::unordered_map<uint32_t, float>& idfMap) override;
  u_int32_t get_seed();
  void set_seed(u_int32_t new_seed);
};

}  // namespace thirdai::utils