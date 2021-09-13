#include <string>
#include <fstream>
#include <iostream>

namespace thirdai::utils {
  class StringLoader {
    public:
    // StringLoader(std::string &filename): _file(filename) {};
    virtual bool loadNextString(std::string &str_buf) {};

    protected:
    // std::ifstream _file;
  };
}