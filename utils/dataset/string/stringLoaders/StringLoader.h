#include <string>

namespace thirdai::utils {
  class StringLoader {
    public:
    virtual void loadNextString(std::string &str_buf);
  };
}