#include "eigen_benchmark.h"
#include <eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h>

void SpatialConvolution(/* Input dimensions: */
                        int batch_size, int input_height, int input_width,
                        int input_depth,
                        /* Filter (kernel) dimensions: */
                        int filter_count, int filter_height, int filter_width) {
  using Benchmark =
      SpatialConvolutionBenchmarksSuite<float, Eigen::DefaultDevice>;
  Eigen::DefaultDevice device = Eigen::DefaultDevice();
  Benchmark benchmark = Benchmark(device);

  typename Benchmark::Dimensions input_dims(batch_size, input_height,
                                            input_width, input_depth);
  typename Benchmark::Dimensions filter_dims(filter_height, filter_width,
                                             input_depth, filter_count);

  benchmark.SpatialConvolution(input_dims, filter_dims);
}

// void SpatialConvolutionBackwardInput(int batch_size, int input_height,
//                                      int input_width, int input_depth,
//                                      /* Filter (kernel) dimensions: */
//                                      int filter_count, int filter_height,
//                                      int filter_width) {

//   using Benchmark =
//       SpatialConvolutionBenchmarksSuite<float, Eigen::DefaultDevice>;
//   Eigen::DefaultDevice device = Eigen::DefaultDevice();
//   Benchmark benchmark = Benchmark(device);

//   typename Benchmark::Dimensions input_dims(batch_size, input_height,
//                                             input_width, input_depth);
//   typename Benchmark::Dimensions filter_dims(filter_height, filter_width,
//                                              input_depth, filter_count);

//   benchmark.SpatialConvolutionBackwardInput(input_dims, filter_dims);
// }

// void SpatialConvolutionBackwardKernel(/* Input dimensions: */
//                                       int batch_size, int input_height,
//                                       int input_width, int input_depth,
//                                       /* Filter (kernel) dimensions: */
//                                       int filter_count, int filter_height,
//                                       int filter_width) {
//   using Benchmark =
//       SpatialConvolutionBenchmarksSuite<float, Eigen::DefaultDevice>;
//   auto device = Eigen::DefaultDevice();
//   auto benchmark = Benchmark(device);


//   typename Benchmark::Dimensions input_dims(batch_size, input_height,
//                                             input_width, input_depth);
//   typename Benchmark::Dimensions filter_dims(filter_height, filter_width,
//                                              input_depth, filter_count);

//   benchmark.SpatialConvolutionBackwardKernel(input_dims, filter_dims);
// }

int main() {
  // SpatialConvolution(32,          // batch size
  //                      56, 56, 64,  // input: height, width, depth
  //                      192, 3, 3);
  SpatialConvolution(0, 0, 0, 0, 0, 0, 0);
}