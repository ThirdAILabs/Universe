#pragma once

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace thirdai::bolt {

struct Batch {
  uint32_t batch_size;
  uint32_t** indices;
  float** values;
  uint32_t* lens;
  uint32_t** labels;
  uint32_t* label_lens;

  Batch()
      : batch_size(0),
        indices(nullptr),
        values(nullptr),
        lens(nullptr),
        labels(nullptr),
        label_lens(nullptr) {}

  explicit Batch(uint64_t _batch_size) : batch_size(_batch_size) {
    indices = new uint32_t*[batch_size];
    values = new float*[batch_size];
    lens = new uint32_t[batch_size];
    labels = new uint32_t*[batch_size];
    label_lens = new uint32_t[batch_size];
  }

  Batch(const Batch& other) = delete;

  Batch(Batch&& other)
      : batch_size(other.batch_size),
        indices(other.indices),
        values(other.values),
        lens(other.lens),
        labels(other.labels),
        label_lens(other.label_lens) {
    other.batch_size = 0;
    other.indices = nullptr;
    other.values = nullptr;
    other.lens = nullptr;
    other.labels = nullptr;
    other.label_lens = nullptr;
  }

  Batch& operator=(const Batch& other) = delete;

  Batch& operator=(Batch&& other) {
    batch_size = other.batch_size;
    indices = other.indices;
    values = other.values;
    lens = other.lens;
    labels = other.labels;
    label_lens = other.label_lens;

    other.batch_size = 0;
    other.indices = nullptr;
    other.values = nullptr;
    other.lens = nullptr;
    other.labels = nullptr;
    other.label_lens = nullptr;
    return *this;
  }

  ~Batch() {
    delete[] indices;
    delete[] values;
    delete[] lens;
    delete[] labels;
    delete[] label_lens;
  }
};

class SvmDataset {
 public:
  SvmDataset(const std::string& filename, uint64_t batch_size);

  const Batch& operator[](uint64_t i) const {
    assert(i <= num_batches);
    return batches[i];
  }

  uint64_t NumBatches() const { return num_batches; }

  uint64_t NumVecs() const { return num_vecs; }

  ~SvmDataset() { delete[] batches; }

  friend std::ostream& operator<<(std::ostream& out, const SvmDataset& data);

 private:
  void ReadDataset(const std::string& filename);

  void CreateBatches();

  std::vector<uint32_t> indices;
  std::vector<float> values;
  std::vector<uint32_t> markers;
  std::vector<uint32_t> labels;
  std::vector<uint32_t> label_markers;

  uint64_t target_batch_size, num_batches, num_vecs;
  Batch* batches;
};

}  // namespace thirdai::bolt