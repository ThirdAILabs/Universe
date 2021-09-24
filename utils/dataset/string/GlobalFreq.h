#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::utils {

enum class FREQ_TYPE { FROM_DATA, DEFAULT };

class GlobalFreq {
  /*
      This class is responsible for calculating (inverse) word/document
     frequencies on a given data corpus.
  */
 private:
  std::unordered_map<std::string, int> _idfMap;
  std::ifstream _file;
  FREQ_TYPE _freq_type;

 public:
  // For now let's assume we're only taking in one file. consider array or list
  // of strings next time. GlobalFreq(std::vector<std::string>& files);
  explicit GlobalFreq(std::string& filename)
      : _file(filename),
        _freq_type(FREQ_TYPE::FROM_DATA){

        };

  GlobalFreq()
      : _freq_type(FREQ_TYPE::DEFAULT){

        };

  int getIdf(std::string& token) {
    switch (_freq_type) {
      case FREQ_TYPE::DEFAULT:
        return 1;
        break;
      default:
        break;
    }
  };  // Should have a default value
  int getTF(std::string& token, std::string& doc);
  int getTokenID(std::string& token);

  // Can have a parallel version of getIdf like:
  void getIdfPar(std::vector<std::string> tokenvec, int* freqs);

  ~GlobalFreq();
};

// static method to initialize global freq to ?? ah ok i think i want a private
// constructor.

// GlobalFreq::GlobalFreq(std::vector<std::string>& files, STRING_TYPE
// load_type) {}

GlobalFreq::~GlobalFreq() {}

}  // namespace thirdai::utils
