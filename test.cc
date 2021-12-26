#include <immintrin.h>
#include <iostream>

int main() {
#ifdef __SSE__
  std::cout << "SSE" << std::endl;
#endif

#ifdef __AVX__
  std::cout << "AVX" << std::endl;
#endif

#ifdef __AVX512F__
  std::cout << "AVX512F" << std::endl;
#endif

  return 0;
}