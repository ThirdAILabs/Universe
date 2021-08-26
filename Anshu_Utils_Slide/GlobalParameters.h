#include <stdint.h>

namespace ThirdAI {

class GlobalParameters {
 private:
  static GlobalParameters _oneTimeParameters;
  static bool _isInitialized;
  GlobalParameters();
  uint32_t _K, _L, _rangePow, _reserviorSize;  // ETC. ETC.
 public:
  static GlobalParameters GetUniqueInstance() {
    if (_isInitialized = false) {
      _oneTimeParameters = GlobalParameters();
      _isInitialized = true;
    }

    return _oneTimeParameters;
  }
  // Getter and setters follow here.
  //
};

}  // namespace ThirdAI
