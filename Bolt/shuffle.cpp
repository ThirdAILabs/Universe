#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

void ShuffleDataset(std::string filename) {
  std::vector<std::string> lines;
  std::ifstream file(filename);

  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }
  file.close();

  std::string outfile;
  auto loc = filename.find(".");
  if (loc != std::string::npos) {
    outfile = filename.substr(0, loc) + "_shuffled" + filename.substr(loc);
  } else {
    outfile += "_shuffled";
  }

  std::random_device rd;

  std::shuffle(lines.begin(), lines.end(), rd);

  std::ofstream output(outfile);

  for (const auto& line : lines) {
    output << line << "\n";
  }
  output.close();

  std::cout << "Shuffled " << lines.size() << " lines from file: '" << filename << "' into '"
            << outfile << "'";
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Invalid args, usage: ./shuffle <path to dataset>" << std::endl;
    return 1;
  }

  ShuffleDataset(argv[1]);

  return 0;
}