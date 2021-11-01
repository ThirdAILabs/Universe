#include "SentenceLoader.h"
namespace thirdai::utils::dataset {

uint32_t SentenceLoader::loadStringsAndLabels(
    std::ifstream& file, uint32_t target_batch_size,
    std::vector<std::string>& loaded_strings,
    std::vector<std::vector<uint32_t>>& loaded_labels) {
  std::string str_buf;
  loaded_strings.reserve(target_batch_size);
  loaded_labels.reserve(target_batch_size);
  while (loaded_strings.size() < target_batch_size &&
         loadNextSentence(file, str_buf)) {
    loaded_strings.push_back(str_buf);
    loaded_labels.push_back(
        std::vector<uint32_t>());  // No labels. Insert empty vectors.
  }
  return loaded_strings.size();
}

bool SentenceLoader::loadNextSentence(std::ifstream& file,
                                      std::string& str_buf) {
  // If line buffer is exhausted, get next line.
  while (_line_buffer.empty()) {
    if (!std::getline(file, _line_buffer)) {
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
}

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
    } else if (!('0' <= c && c <= '9') && c != '.' && c != ' ' &&
               !('a' <= c && c <= 'z')) {
      c = '~';
    }
  }

  // Mark spaces that come after periods to be removed.
  // Mark periods or spaces that come after other periods or spaces to be
  // removed (do not accept consecutive spaces or periods).
  char last_c = '~';
  for (auto& c : line_buffer) {
    if (((c == '.' || c == ' ') && c == last_c) ||
        (c == ' ' && last_c == '.')) {
      last_c = c;
      c = '~';
    } else if (c != '~') {
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
  line_buffer.erase(std::remove(line_buffer.begin(), line_buffer.end(), '~'),
                    line_buffer.end());
}
}  // namespace thirdai::utils::dataset