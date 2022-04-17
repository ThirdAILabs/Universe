#include <wrappers/src/EigenDenseWrapper.h>
#include <chrono>
#include <iostream>
#include <vector>

using namespace std::chrono;

int main() {
  uint32_t num_loops = 1000;
  std::vector<Eigen::MatrixXf> tensors(num_loops);
  for (int i = 0; i < num_loops; i++) {
    tensors[i] = Eigen::MatrixXf::Random(100000 / num_loops, 100);
  }

  for (int i = 0; i < 100; i++) {
    Eigen::MatrixXf t_2 = Eigen::MatrixXf::Random(100, 50);

    uint64_t start =
        duration_cast<milliseconds>(system_clock::now().time_since_epoch())
            .count();

    float test = 0;
#pragma omp parallel for
    for (int i = 0; i < num_loops; i++) {
      Eigen::MatrixXf t_3 = tensors.at(i) * t_2;
      test += t_3(0, 0);
    }
    std::cout << duration_cast<milliseconds>(
                     system_clock::now().time_since_epoch())
                         .count() -
                     start
              << std::endl;
  }
}