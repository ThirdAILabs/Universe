#include <stdexcept>

namespace thirdai::utils {

class NotImplemented : public std::logic_error {
 public:
  NotImplemented() : std::logic_error("Function not yet implemented"){};
};

}  // namespace thirdai::utils