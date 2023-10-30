#include "SeismicLabels.h"
#include <hashing/src/MurmurHash.h>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt::seismic {

std::vector<uint32_t> seismicLabelsFromMetadata(
    const SubcubeMetadata& subcube_metadata, const Shape& subcube_shape,
    size_t label_cube_dim, size_t max_label) {
  return seismicLabels(
      std::get<0>(subcube_metadata), std::get<1>(subcube_metadata),
      std::get<2>(subcube_metadata), std::get<3>(subcube_metadata),
      subcube_shape, label_cube_dim, max_label);
}

std::vector<uint32_t> seismicLabels(const std::string& volume, size_t x_coord,
                                    size_t y_coord, size_t z_coord,
                                    const Shape& subcube_shape,
                                    size_t label_cube_dim, size_t max_label) {
  // We use the label_cube_dim as the seed so that if we want to get multiple
  // sets of labels at different granularities we can.
  uint32_t trace_seed =
      hashing::MurmurHash(volume.data(), volume.size(), label_cube_dim);

  std::vector<uint32_t> labels;

  size_t start_x_label = x_coord / label_cube_dim;
  size_t start_y_label = y_coord / label_cube_dim;
  size_t start_z_label = z_coord / label_cube_dim;

  auto [dim_x, dim_y, dim_z] = subcube_shape;

  // We take max(dim / label_cube_dim, 1) because if the subcube has a shape
  // like  (1, 10, 10), and the label cube dim is 5, then we still want to
  // compute labels in the x axis, there will just only be one label in the
  // x-axis, and 2 in each of the y and z.
  for (size_t x = 0; x < std::max<size_t>(dim_x / label_cube_dim, 1); x++) {
    for (size_t y = 0; y < std::max<size_t>(dim_y / label_cube_dim, 1); y++) {
      for (size_t z = 0; z < std::max<size_t>(dim_z / label_cube_dim, 1); z++) {
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

}  // namespace thirdai::bolt::seismic