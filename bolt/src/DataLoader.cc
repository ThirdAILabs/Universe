#include "DataLoader.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace thirdai::bolt {

SvmDataset::SvmDataset(const std::string& filename, uint64_t _batch_size)
    : target_batch_size(_batch_size), num_vecs(0) {
  auto start = std::chrono::high_resolution_clock::now();

  ReadDataset(filename);

  num_batches = (num_vecs + target_batch_size - 1) / target_batch_size;

  batches = new Batch[num_batches];
  for (uint64_t i = 0; i < num_batches; i++) {
    uint32_t batch_size =
        std::min(target_batch_size, num_vecs - i * target_batch_size);
    batches[i] = Batch(batch_size);
  }

  CreateBatches();

  auto end = std::chrono::high_resolution_clock::now();

  std::cout
      << "\033[1;36mRead " << num_vecs << " vectors from '" << filename
      << "' and created " << num_batches << " batches in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds \033[0m" << std::endl;
}

void SvmDataset::ReadDataset(const std::string& filename) {
  std::ifstream file(filename);
  if (file.bad() || file.fail() || !file.good() || !file.is_open()) {
    throw std::runtime_error("Unable to open file '" + filename + "'");
  }
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream stream(line);

    label_markers.push_back(labels.size());
    std::string labelstr;
    stream >> labelstr;
    size_t pos;
    while ((pos = labelstr.find(',')) != std::string::npos) {
      labels.push_back(atoi(labelstr.substr(0, pos).c_str()));
      labelstr = labelstr.substr(pos + 1);
    }
    labels.push_back(atoi(labelstr.c_str()));

    markers.push_back(indices.size());
    std::string nonzero;
    while (stream >> nonzero) {
      pos = nonzero.find(':');
      indices.push_back(atoi(nonzero.substr(0, pos).c_str()));
      values.push_back(atof(nonzero.substr(pos + 1).c_str()));
    }
  }

  label_markers.push_back(labels.size());
  markers.push_back(indices.size());

  num_vecs = markers.size() - 1;

  file.close();
}

void SvmDataset::CreateBatches() {
  uint32_t* indices_ptr = indices.data();
  float* values_ptr = values.data();
  uint32_t* labels_ptr = labels.data();

  for (uint64_t batch = 0; batch < num_batches; batch++) {
    for (uint64_t n = 0; n < batches[batch].batch_size; n++) {
      uint64_t indx = batch * target_batch_size + n;
      batches[batch].indices[n] = indices_ptr + markers[indx];
      batches[batch].values[n] = values_ptr + markers[indx];
      batches[batch].lens[n] = markers[indx + 1] - markers[indx];

      batches[batch].labels[n] = labels_ptr + label_markers[indx];
      batches[batch].label_lens[n] =
          label_markers[indx + 1] - label_markers[indx];
    }
  }
}

std::ostream& operator<<(std::ostream& out, const SvmDataset& data) {
  for (uint64_t batch = 0; batch < data.num_batches; batch++) {
    out << "Batch: " << batch << std::endl;
    for (uint64_t i = 0; i < data[batch].batch_size; i++) {
      out << "Labels:";
      for (uint32_t j = 0; j < data[batch].label_lens[i]; j++) {
        out << " " << data[batch].labels[i][j];
      }
      out << std::endl << "Vector:";
      for (uint32_t j = 0; j < data[batch].lens[i]; j++) {
        out << " (" << data[batch].indices[i][j] << ", "
            << data[batch].values[i][j] << ")";
      }
      out << std::endl;
    }
    out << std::endl;
  }
  return out;
}

}  // namespace thirdai::bolt
