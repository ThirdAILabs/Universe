#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace thirdai::bolt::seismic {

using Shape = std::tuple<size_t, size_t, size_t>;

using SubcubeMetadata = std::tuple<std::string, size_t, size_t, size_t>;

std::vector<uint32_t> seismicLabelsFromMetadata(
    const SubcubeMetadata& subcube_metadata, const Shape& subcube_shape,
    size_t label_cube_dim, size_t max_label);

std::vector<uint32_t> seismicLabels(const std::string& volume, size_t x_coord,
                                    size_t y_coord, size_t z_coord,
                                    const Shape& subcube_shape,
                                    size_t label_cube_dim, size_t max_label);

}  // namespace thirdai::bolt::seismic