#include <iostream>
#include <fstream>
#include "StringLoader.h"

namespace thirdai::utils {
  class SentenceLoader : public StringLoader {
    public:
    SentenceLoader(std::string &filename): _file(filename) {};
    virtual bool loadNextString(std::string &str_buf) {
      while (_lb_idx == _line_buffer.length()) {
        if (!std::getline(_file, _line_buffer)) {
          return false;
        } // Need to check whether EOF?
        _lb_idx = 0;
        cleanUpLineBuffer();
      }
      size_t start_lb_idx = _lb_idx;
      bool not_sentence_delimiter = true;
      for (_lb_idx; _lb_idx < _line_buffer.length() && not_sentence_delimiter; _lb_idx++) {
        not_sentence_delimiter = notSentenceDelimiter(_line_buffer[_lb_idx], str_buf);
      }
      str_buf = _line_buffer.substr(start_lb_idx, _lb_idx);
      if (!not_sentence_delimiter) {
        _lb_idx++;
      }
      return true;
    };
    private:
    std::ifstream _file;
    std::string _line_buffer = "";
    size_t _lb_idx = 0;

    void cleanUpLineBuffer() {};
    bool notSentenceDelimiter(char c, std::string& str) {
      return c != '.' && c != '?' && c != '!';
    }
  };
}