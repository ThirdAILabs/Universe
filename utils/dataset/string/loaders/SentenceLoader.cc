#include "SentenceLoader.h"
namespace thirdai::utils {

bool SentenceLoader::loadNextString(std::string& str_buf) {
  // If line buffer is exhausted, get next line.
  while (_line_buffer.empty()) {
    if (!getNextLine(_line_buffer)) {
      return false;
    }
    _lb_idx = 0;
    cleanUpLineBuffer(_line_buffer);
  }

  // Find the next sentence
  size_t pos = _line_buffer.find('.');
  if (pos != std::string::npos) {
    str_buf = _line_buffer.substr(0, pos);
    _line_buffer = _line_buffer.substr(pos + 1);
  } else {
    str_buf = _line_buffer;
    _line_buffer = "";
  }
  return true;
};

void SentenceLoader::cleanUpLineBuffer(std::string& line_buffer) {
    // Turn all whitespaces into space.
    // Turn all uppercase characters into lowercase.
    // Turn all sentence ending punctuation marks to periods.
    // Remove any character that is not a letter, a number, 
    // or a sentence sending punctuation mark.
    for (auto& c : line_buffer) {
      if (isspace(c) && c != ' ') {
        c = ' ';
      } else if ('A' <= c && c <= 'Z') {
        c = tolower(c);
      } else if (c == '?' || c == '!') {
        c = '.';
      } else if (!('0' <= c && c <= '9') || c != '.') {
        c = '~';
      }
    }
    
    // Mark spaces that come after periods to be removed.
    // Mark periods or spaces that come after other periods or spaces to be removed 
    // (do not accept consecutive spaces or periods).
    char last_c = '~';
    for (auto& c : line_buffer) {
      if (((c == '.' || c == ' ') && c == last_c) || (c == ' ' && last_c == '.')) {
        last_c = c;
        c = '~';
      } else {
        last_c = c;
      }
    }

    // Mark periods or spaces before the first letter or number to be removed.
    for (auto& c : line_buffer) {
      if (c == '.' || c == ' ') {
        c = '~';
      } else {
        break;
      }
    }

    // Mark periods or spaces after the last letter or number to be removed.
    for (auto it = line_buffer.rbegin(); it != line_buffer.rend(); it++) {
      if (*it == '.' || *it == ' ') {
        *it = '~';
      } else {
        break;
      }
    }
    
    // Remove everything marked to be removed.
    line_buffer.erase(std::remove(line_buffer.begin(), line_buffer.end(), '~'), line_buffer.end());

  };

  
  bool SentenceLoader::getNextLine(std::string& next_line_buf) {
    // Make sure that file is open, file is not bad, file is not exhausted.
    while (_queue_idx < _filename_queue.size()) {
      // make sure file is open and good.
      while (!_file.is_open() && _queue_idx < _filename_queue.size()) {
        _file.open(_filename_queue[_queue_idx]);
        if (_file.bad() || _file.fail() || !_file.good() || !_file.is_open()) {
          _file.close();
          _queue_idx++;
        } 
      }
      if (!_file.is_open()) {
        return false;
      }
      // if file is exhausted, stay in the loop, otherwise return true.
      if (!std::getline(_file, next_line_buf)) {
        _file.close();
        _queue_idx++;
      } else {
        return true;
      }
    }
    return false;
  }
}  // namespace thirdai::utils