#include <chrono>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "Schema.h"
#include "Text.h"
#include "FeatureHashing.h"
#include "Number.h"
#include "DynamicCounts.h"
#include "NumericalLabel.h"

using thirdai::schema::DataLoader;
using thirdai::schema::ABlockBuilder;
using thirdai::schema::DynamicCountsBlock;
using thirdai::schema::Window;
using thirdai::schema::NumericalLabelBlock;

int main(int argc, char** argv) {
  (void) argc;
  (void) argv;

  uint32_t n_lines = 10;
  std::vector<std::string> lines(n_lines);
  for (size_t i = 0; i < n_lines; ++i) {
    if (!(i % n_lines / 10)) {
      std::cout << '.';
    }
    std::stringstream ss;
    uint32_t day = 1 + i * 25 / n_lines ;
    std::string trailing_0 = day < 10 ? "0" : "";
    ss << "0,2014-Feb-" << trailing_0 << day << "," << i % 5;
    lines.push_back(ss.str());
  }
  std::cout << "Generated mock lines" << std::endl;
  
  std::vector<std::shared_ptr<ABlockBuilder>> schema;
  std::vector<Window> window_configs;
  window_configs.push_back(Window(6, 12));
  uint32_t id_col = 0;
  uint32_t timestamp_col = 1;

  uint32_t target_col = 2;
  std::string timestamp_fmt = "%Y-%b-%d";


///////////////////////////


  auto label_block = NumericalLabelBlock::Builder(2);

  auto dynamic_counts_block = DynamicCountsBlock::Builder(
    id_col, 
    timestamp_col, 
    target_col,
    window_configs,
    window_configs,
    timestamp_fmt);
  
  // schema.push_back(label_block);
  schema.push_back(dynamic_counts_block);
  uint32_t batch_size = 256;
  DataLoader data(schema, batch_size);

  auto start = std::chrono::high_resolution_clock::now();
  for (const auto& line : lines) {
    data.consumeCSVLine(line);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Done in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds." << std::endl;



/////////////////////////
  auto dataset = data.exportDataset();
  std::cout << "size of data is " << dataset.len() << std::endl;
  std::cout << dataset.at(0).getBatchSize() << std::endl;
  for (size_t i = 0; i < dataset.at(0).at(19).length(); ++i) {
    std::cout << "(" << dataset.at(0).at(19)._indices[i] << ", " << dataset.at(0).at(19)._values[i] << ")" << std::endl;
  }
  for (const auto& label : dataset.at(0).labels(19)) {
    std::cout << label << ", " << std::endl;
  }

  return 0;
}