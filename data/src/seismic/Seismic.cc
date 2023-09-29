#include "Seismic.h"
#include <hashing/src/MurmurHash.h>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace thirdai::data {

constexpr uint32_t SEED = 102942;

std::vector<uint32_t> seismicLabels(const std::string& trace, size_t x_coord,
                                    size_t y_coord, size_t z_coord,
                                    size_t subcube_dim, size_t label_cube_dim,
                                    size_t max_label) {
  if ((subcube_dim % label_cube_dim) != 0) {
    throw std::invalid_argument(
        "Expected subcube_dim to be a multiple of label_cube_dim.");
  }
  uint32_t trace_seed = hashing::MurmurHash(trace.data(), trace.size(), SEED);

  std::vector<uint32_t> labels;

  size_t start_x_label = x_coord / label_cube_dim;
  size_t start_y_label = y_coord / label_cube_dim;
  size_t start_z_label = z_coord / label_cube_dim;
  size_t label_cubes_per_subcube = subcube_dim / label_cube_dim;

  for (size_t x = 0; x < label_cubes_per_subcube; x++) {
    for (size_t y = 0; y < label_cubes_per_subcube; y++) {
      for (size_t z = 0; z < label_cubes_per_subcube; z++) {
        size_t label_coords[3] = {start_x_label + x, start_y_label + y,
                                  start_z_label + z};
        uint32_t hash =
            hashing::MurmurHash(reinterpret_cast<const char*>(&label_coords),
                                3 * sizeof(size_t), trace_seed);
        labels.push_back(hash % max_label);
      }
    }
  }

  return labels;
}

}  // namespace thirdai::data