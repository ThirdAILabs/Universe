#include "Constraints.h"

namespace thirdai::search::ndb {

bool matches(const QueryConstraints& constraints, const MetadataMap& metadata) {
  // NOLINTNEXTLINE (clang tidy wants std::all_of)
  for (const auto& [key, constraint] : constraints) {
    if (!metadata.count(key)) {
      return false;
    }
    if (!constraint->matches(metadata.at(key))) {
      return false;
    }
  }

  return true;
}

}  // namespace thirdai::search::ndb