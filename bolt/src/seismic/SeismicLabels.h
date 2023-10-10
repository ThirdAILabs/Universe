#pragma once

#include <string>
#include <vector>

namespace thirdai::bolt::seismic {

struct SubcubeMetadata {
  SubcubeMetadata(std::string volume, size_t x, size_t y, size_t z)
      : volume_name(std::move(volume)), x_coord(x), y_coord(y), z_coord(z) {}

  std::string volume_name;
  size_t x_coord;
  size_t y_coord;
  size_t z_coord;
};

std::vector<uint32_t> seismicLabelsFromMetadata(
    const SubcubeMetadata& subcube_metadata, size_t subcube_dim,
    size_t label_cube_dim, size_t max_label);

std::vector<uint32_t> seismicLabels(const std::string& volume, size_t x_coord,
                                    size_t y_coord, size_t z_coord,
                                    size_t subcube_shape, size_t label_cube_shape,
                                    size_t max_label);

}  // namespace thirdai::bolt::seismic